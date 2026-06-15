import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

# UMI Framework components
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

# Dynamixel SDK
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead

# XC330-T181-T Control Table Addresses
ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
ADDR_PRESENT_LOAD           = 126 # [2 bytes] Start Reading Here
# ADDR_PRESENT_VELOCITY     = 128 # [4 bytes]
ADDR_PRESENT_POSITION       = 132 # [4 bytes] End Reading Here

# Data Lengths
LEN_GOAL_POSITION           = 4
LEN_PRESENT_POSITION        = 4
LEN_PRESENT_LOAD            = 2
# Read 10 bytes to cover Load(126) to Position(132+4)
LEN_PRESENT_FULL_BLOCK      = 10 

PROTOCOL_VERSION            = 2.0
TORQUE_ENABLE               = 1
TORQUE_DISABLE              = 0

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class DynamixelController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            device_name: str,
            baudrate: int,
            dxl_ids: list,
            gripper_max_width_mm: float,  # This will receive 80.0
            motor_positions_open: list,   # This will receive recalculated ticks for 80mm
            motor_positions_closed: list,
            frequency=30,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            verbose=False
            ):
        super().__init__(name="DynamixelController")
        
        self.device_name = device_name
        self.baudrate = baudrate
        self.dxl_ids = dxl_ids
        self.gripper_max_width_mm = gripper_max_width_mm
        self.motor_positions_open = np.array(motor_positions_open)
        self.motor_positions_closed = np.array(motor_positions_closed)
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose

        # Validate parameters
        assert len(self.dxl_ids) == 2
        
        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        state_example = {
            'gripper_position_mm': 5.0,
            'gripper_velocity_mm_s': 0.0,
            'gripper_force': 0.0,
            f'dxl_{self.dxl_ids[0]}_position': 0,
            f'dxl_{self.dxl_ids[1]}_position': 0,
            f'dxl_{self.dxl_ids[0]}_load': 0,
            f'dxl_{self.dxl_ids[1]}_load': 0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=state_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()

    def _mm_to_motor_positions(self, gripper_width_mm: float) -> np.ndarray:
        # If max_width is 80, this ratio maps 0~80mm to 0.0~1.0
        width_ratio = np.clip(gripper_width_mm / self.gripper_max_width_mm, 0.0, 1.0)
        target_positions = self.motor_positions_closed * (1 - width_ratio) + self.motor_positions_open * width_ratio
        return target_positions.astype(np.int32)

    def _motor_positions_to_mm(self, motor_positions: np.ndarray) -> float:
        pos_diff = self.motor_positions_open - self.motor_positions_closed
        pos_diff[pos_diff == 0] = 1 
        current_pos_ratio = (motor_positions - self.motor_positions_closed) / pos_diff
        avg_ratio = np.mean(current_pos_ratio)
        gripper_width_mm = self.gripper_max_width_mm * avg_ratio
        return gripper_width_mm

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[DynamixelController] Process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def run(self):
        portHandler = PortHandler(self.device_name)
        packetHandler = PacketHandler(PROTOCOL_VERSION)
        
        groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        # FIX: Read from Load(126) to Position(132) inclusive (10 bytes)
        groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_LOAD, LEN_PRESENT_FULL_BLOCK)

        try:
            if not portHandler.openPort():
                print(f"[ERROR] Failed to open port: {self.device_name}")
                return
            if not portHandler.setBaudRate(self.baudrate):
                print(f"[ERROR] Failed to set baudrate: {self.baudrate}")
                return
            
            for dxl_id in self.dxl_ids:
                if not groupSyncRead.addParam(dxl_id):
                    print(f"[ERROR] Failed to add SyncRead param for ID {dxl_id}")

            # Initial move to 50%
            initial_pos_ticks = self._mm_to_motor_positions(self.gripper_max_width_mm / 2.0)
            for i, dxl_id in enumerate(self.dxl_ids):
                packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, int(initial_pos_ticks[i]))

            time.sleep(1.0) 

            # Initialize interpolator
            curr_pos_mm = self._motor_positions_to_mm(initial_pos_ticks)
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=np.array([[curr_pos_mm,0,0,0,0,0]])
            )
            
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            
            while keep_running:
                t_now = time.monotonic()
                target_pos_mm = pose_interp(t_now)[0]
                target_pos_ticks = self._mm_to_motor_positions(target_pos_mm)

                # SyncWrite
                groupSyncWrite.clearParam()
                for i, dxl_id in enumerate(self.dxl_ids):
                    param_goal_position = [
                        (target_pos_ticks[i] >> 0) & 0xFF,
                        (target_pos_ticks[i] >> 8) & 0xFF,
                        (target_pos_ticks[i] >> 16) & 0xFF,
                        (target_pos_ticks[i] >> 24) & 0xFF
                    ]
                    groupSyncWrite.addParam(dxl_id, param_goal_position)
                groupSyncWrite.txPacket()

                # SyncRead
                groupSyncRead.txRxPacket()
                
                t_recv = time.time()
                state = dict()
                motor_positions = np.zeros(2, dtype=np.int32)
                motor_loads = np.zeros(2, dtype=np.int16)

                for i, dxl_id in enumerate(self.dxl_ids):
                    # 1. Position (ADDR 132)
                    if groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                        pos = groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                        if pos > 0x7FFFFFFF: pos -= 4294967296
                        motor_positions[i] = pos

                    # 2. Load (ADDR 126)
                    if groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_LOAD, LEN_PRESENT_LOAD):
                        load = groupSyncRead.getData(dxl_id, ADDR_PRESENT_LOAD, LEN_PRESENT_LOAD)
                        if load > 0x7FFF: load -= 65536
                        motor_loads[i] = load

                    state[f'dxl_{dxl_id}_position'] = motor_positions[i]

                    state[f'dxl_{dxl_id}_load'] = motor_loads[i]
                
                current_pos_mm = self._motor_positions_to_mm(motor_positions)
                
                dt = 1 / self.frequency
                prev_pos_mm = pose_interp(t_now - dt)[0]
                velocity_mm_s = (current_pos_mm - prev_pos_mm) / dt

                state['gripper_position_mm'] = current_pos_mm
                state['gripper_velocity_mm_s'] = velocity_mm_s
                state['gripper_force'] = np.mean(np.abs(motor_loads))
                state['gripper_receive_timestamp'] = t_recv
                state['gripper_timestamp'] = t_recv - self.receive_latency

                self.ring_buffer.put(state)

                # Process Commands
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                
                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command['cmd']
                    
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = time.monotonic() - time.time() + command['target_time']
                        
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=np.array([target_pos, 0,0,0,0,0]),
                            time=target_time,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                
                if not keep_running:
                    break
                    
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
                
        finally:
            for dxl_id in self.dxl_ids:
                packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            portHandler.closePort()
            self.ready_event.set()
            if self.verbose:
                print(f"[DynamixelController] Disconnected: {self.device_name}")