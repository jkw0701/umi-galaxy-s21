import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing.managers import SharedMemoryManager
from threading import Thread
import traceback

try:
    from dynamixel_sdk import *
except ImportError:
    print("Dynamixel SDK not found. Please install it first.")
    print("pip install dynamixel-sdk")
    exit()


class DynamixelController:
    """
    XL330 전용 전류 센싱 기능이 포함된 다이나믹셀 그리퍼 컨트롤러
    별도 프로세스에서 실행되며 공유 메모리를 통해 통신합니다.
    """
    
    # XL330 다이나믹셀 레지스터 주소
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_PRESENT_CURRENT = 126  # XL330: 입력 전원 전류 측정
    ADDR_CURRENT_LIMIT = 38
    ADDR_PROFILE_VELOCITY = 112
    
    # 프로토콜 버전
    PROTOCOL_VERSION = 2.0
    
    def __init__(self, 
                 shm_manager: SharedMemoryManager,
                 device_name: str = '/dev/ttyUSB0',
                 baudrate: int = 57600,
                 dxl_ids: list = [10, 11],
                 gripper_max_width_mm: float = 55.0,
                 motor_positions_open: list = [2048, 2048],
                 motor_positions_closed: list = [1024, 3072],
                 current_unit_to_ma: float = 1.0,  # XL330: 1 unit = 약 1mA
                 max_current_limit_ma: float = 500.0):  # XL330 안전 전류 제한
        """
        XL330용 DynamixelController 초기화
        """
        self.shm_manager = shm_manager
        self.device_name = device_name
        self.baudrate = baudrate
        self.dxl_ids = dxl_ids
        self.gripper_max_width_mm = gripper_max_width_mm
        self.motor_positions_open = motor_positions_open
        self.motor_positions_closed = motor_positions_closed
        self.current_unit_to_ma = current_unit_to_ma
        self.max_current_limit_ma = max_current_limit_ma
        
        # 다이나믹셀 SDK 객체들
        self.portHandler = None
        self.packetHandler = None
        
        # 프로세스 관련
        self.process = None
        self.stop_event = None
        self.ready_event = None
        
        # 공유 메모리 생성
        self._setup_shared_memory()
        
        # 위치-거리 변환을 위한 계수 계산
        self._calculate_conversion_factors()
    
    def _setup_shared_memory(self):
        """공유 메모리 설정"""
        # 명령 공유 메모리 (target_position, target_time)
        self.command_shm = self.shm_manager.SharedMemory(size=16)  # 2 * 8 bytes (double)
        self.command_array = np.ndarray((2,), dtype=np.float64, buffer=self.command_shm.buf)
        self.command_array[0] = 0.0  # target_position_mm
        self.command_array[1] = 0.0  # target_time
        
        # 상태 공유 메모리 - XL330 전용 확장
        self.state_shm = self.shm_manager.SharedMemory(size=80)  # 10 * 8 bytes
        self.state_array = np.ndarray((10,), dtype=np.float64, buffer=self.state_shm.buf)
        # [current_position_mm, current_velocity, motor1_current_ma, motor2_current_ma, 
        #  max_load_current, total_load_current, direction_consistent, timestamp, status, reserved]
        self.state_array[:] = 0.0
    
    def _calculate_conversion_factors(self):
        """위치값과 실제 거리 간 변환 계수 계산"""
        pos_diff_open = abs(self.motor_positions_open[1] - self.motor_positions_open[0])
        pos_diff_closed = abs(self.motor_positions_closed[1] - self.motor_positions_closed[0])
        
        avg_pos_diff_per_mm = (pos_diff_open + pos_diff_closed) / (2 * self.gripper_max_width_mm)
        self.pos_to_mm_factor = 1.0 / avg_pos_diff_per_mm if avg_pos_diff_per_mm != 0 else 1.0
    
    def start(self):
        """컨트롤러 프로세스 시작"""
        if self.process is not None:
            return
        
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        
        self.process = Process(target=self._controller_process)
        self.process.start()
    
    def start_wait(self, timeout: float = 5.0):
        """컨트롤러가 준비될 때까지 대기"""
        if self.ready_event is None:
            return False
        return self.ready_event.wait(timeout)
    
    def stop(self):
        """컨트롤러 프로세스 정지"""
        if self.stop_event is not None:
            self.stop_event.set()
    
    def join(self, timeout: float = 5.0):
        """프로세스 종료 대기"""
        if self.process is not None:
            self.process.join(timeout)
            self.process = None
    
    def schedule_waypoint(self, pos: float, target_time: float):
        """목표 위치 설정"""
        self.command_array[0] = pos
        self.command_array[1] = target_time
    
    def get_state(self) -> dict:
        """XL330 전용 상태 반환"""
        return {
            'gripper_position_mm': self.state_array[0],
            'gripper_velocity': self.state_array[1],
            'motor_currents_ma': [self.state_array[2], self.state_array[3]],
            'max_load_current_ma': self.state_array[4],  # 절댓값 최대 부하
            'total_load_current_ma': self.state_array[5],  # 총 부하 전류
            'direction_consistent': bool(self.state_array[6]),  # 방향 일관성
            'timestamp': self.state_array[7],
            'status': int(self.state_array[8])
        }
    
    def get_current(self) -> list:
        """현재 전류값 반환 (호환성용)"""
        return [self.state_array[2], self.state_array[3]]
    
    def _controller_process(self):
        """메인 컨트롤러 프로세스"""
        try:
            # 다이나믹셀 초기화
            if not self._initialize_dynamixel():
                print("Failed to initialize Dynamixel")
                return
            
            # XL330 전류 센싱 설정
            self._setup_xl330_current_sensing()
            
            # 준비 완료 신호
            self.ready_event.set()
            print("XL330 Dynamixel controller ready")
            
            # 메인 루프
            self._xl330_control_loop()
            
        except Exception as e:
            print(f"Controller process error: {e}")
            traceback.print_exc()
        finally:
            self._cleanup_dynamixel()
    
    def _initialize_dynamixel(self) -> bool:
        """XL330 다이나믹셀 초기화"""
        try:
            # 포트 및 패킷 핸들러 생성
            self.portHandler = PortHandler(self.device_name)
            self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
            
            # 포트 열기
            if not self.portHandler.openPort():
                print(f"Failed to open port {self.device_name}")
                return False
            
            # 통신 속도 설정
            if not self.portHandler.setBaudRate(self.baudrate):
                print(f"Failed to set baudrate to {self.baudrate}")
                return False
            
            # XL330 모터별 초기 설정
            for dxl_id in self.dxl_ids:
                # 토크 활성화
                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 1)
                
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"Failed to enable torque for XL330 motor {dxl_id}")
                    return False
                
                # XL330에 적합한 프로파일 속도 설정
                dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_PROFILE_VELOCITY, 50)  # 더 부드러운 동작
            
            print("XL330 initialization successful")
            return True
            
        except Exception as e:
            print(f"XL330 initialization error: {e}")
            return False
    
    def _setup_xl330_current_sensing(self):
        """XL330 전용 전류 센싱 설정"""
        try:
            for dxl_id in self.dxl_ids:
                # XL330에 맞는 전류 제한 설정
                current_limit_units = int(self.max_current_limit_ma / self.current_unit_to_ma)
                dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_CURRENT_LIMIT, current_limit_units)
                
                if dxl_comm_result == COMM_SUCCESS:
                    print(f"Set XL330 current limit for motor {dxl_id}: {self.max_current_limit_ma}mA")
                else:
                    print(f"Warning: Failed to set current limit for XL330 motor {dxl_id}")
                    
        except Exception as e:
            print(f"XL330 current sensing setup error: {e}")
    
    def _read_xl330_current_values(self) -> list:
        """XL330용 입력 전원 전류 읽기"""
        current_values = []
        
        try:
            for dxl_id in self.dxl_ids:
                # XL330 Present Current 읽기 (입력 전원 전류)
                present_current, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_PRESENT_CURRENT)
                
                if dxl_comm_result == COMM_SUCCESS:
                    # 2의 보수 형태 변환
                    if present_current > 32767:
                        present_current = present_current - 65536
                    
                    # XL330: 1 unit = 약 1mA (입력 전원 전류)
                    current_ma = present_current * self.current_unit_to_ma
                    current_values.append(current_ma)
                else:
                    current_values.append(0.0)
                    
        except Exception as e:
            print(f"Error reading XL330 current values: {e}")
            current_values = [0.0] * len(self.dxl_ids)
        
        return current_values
    
    def _calculate_xl330_load_analysis(self, current_values: list) -> dict:
        """XL330의 입력 전원 전류를 기반으로 한 부하 분석"""
        if len(current_values) < 2:
            return {
                'max_load': 0, 
                'total_load': 0, 
                'direction_consistent': False,
                'raw_currents': [0, 0],
                'abs_currents': [0, 0]
            }
        
        # 절댓값으로 실제 부하 계산
        abs_currents = [abs(current) for current in current_values]
        max_load = max(abs_currents)
        total_load = sum(abs_currents)
        
        # XL330 그리퍼: 두 모터가 반대 방향으로 움직여야 정상
        direction_consistent = (current_values[0] * current_values[1]) <= 0
        
        return {
            'max_load': max_load,
            'total_load': total_load,
            'direction_consistent': direction_consistent,
            'raw_currents': current_values,
            'abs_currents': abs_currents
        }
    
    def _read_positions(self) -> list:
        """모든 모터의 현재 위치 읽기"""
        positions = []
        
        try:
            for dxl_id in self.dxl_ids:
                present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION)
                
                if dxl_comm_result == COMM_SUCCESS:
                    positions.append(present_position)
                else:
                    positions.append(0)
                    
        except Exception as e:
            print(f"Error reading positions: {e}")
            positions = [0] * len(self.dxl_ids)
        
        return positions
    
    def _positions_to_gripper_width(self, positions: list) -> float:
        """모터 위치값을 그리퍼 너비(mm)로 변환"""
        if len(positions) < 2:
            return 0.0
        
        pos_diff = abs(positions[1] - positions[0])
        open_diff = abs(self.motor_positions_open[1] - self.motor_positions_open[0])
        closed_diff = abs(self.motor_positions_closed[1] - self.motor_positions_closed[0])
        
        if open_diff != closed_diff:
            normalized = (pos_diff - closed_diff) / (open_diff - closed_diff)
        else:
            normalized = 0.5
        
        normalized = max(0.0, min(1.0, normalized))
        gripper_width = normalized * self.gripper_max_width_mm
        
        return gripper_width
    
    def _gripper_width_to_positions(self, width_mm: float) -> list:
        """그리퍼 너비(mm)를 모터 목표 위치로 변환"""
        normalized = width_mm / self.gripper_max_width_mm
        normalized = max(0.0, min(1.0, normalized))
        
        target_positions = []
        for i in range(len(self.dxl_ids)):
            open_pos = self.motor_positions_open[i]
            closed_pos = self.motor_positions_closed[i]
            
            target_pos = closed_pos + normalized * (open_pos - closed_pos)
            target_positions.append(int(target_pos))
        
        return target_positions
    
    def _set_goal_positions(self, positions: list):
        """모터 목표 위치 설정"""
        try:
            for i, dxl_id in enumerate(self.dxl_ids):
                if i < len(positions):
                    dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                        self.portHandler, dxl_id, self.ADDR_GOAL_POSITION, positions[i])
                    
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"Failed to set goal position for XL330 motor {dxl_id}")
                        
        except Exception as e:
            print(f"Error setting goal positions: {e}")
    
    def _xl330_control_loop(self):
        """XL330 전용 제어 루프"""
        last_target_time = 0.0
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # 명령 읽기
                target_pos_mm = self.command_array[0]
                target_time = self.command_array[1]
                
                # 새 명령이 있으면 실행
                if target_time > last_target_time and target_time <= current_time + 0.5:
                    target_positions = self._gripper_width_to_positions(target_pos_mm)
                    self._set_goal_positions(target_positions)
                    last_target_time = target_time
                
                # XL330 상태 읽기
                current_positions = self._read_positions()
                current_width = self._positions_to_gripper_width(current_positions)
                current_values = self._read_xl330_current_values()
                
                # XL330용 부하 분석
                load_analysis = self._calculate_xl330_load_analysis(current_values)
                
                # XL330 전용 상태 업데이트
                self.state_array[0] = current_width  # gripper_position_mm
                self.state_array[1] = 0.0  # gripper_velocity (추후 구현)
                self.state_array[2] = current_values[0] if len(current_values) > 0 else 0.0  # motor1 current
                self.state_array[3] = current_values[1] if len(current_values) > 1 else 0.0  # motor2 current
                self.state_array[4] = load_analysis['max_load']  # 최대 부하 전류 (절댓값)
                self.state_array[5] = load_analysis['total_load']  # 총 부하 전류
                self.state_array[6] = 1.0 if load_analysis['direction_consistent'] else 0.0  # 방향 일관성
                self.state_array[7] = current_time  # timestamp
                self.state_array[8] = 1.0  # status (1 = running)
                
                # XL330에 적합한 주기 (입력 전원 전류는 상대적으로 안정적)
                time.sleep(0.02)  # 50Hz
                
            except Exception as e:
                print(f"XL330 control loop error: {e}")
                time.sleep(0.1)
    
    def _cleanup_dynamixel(self):
        """XL330 다이나믹셀 정리"""
        try:
            if self.portHandler and self.packetHandler:
                # 모든 XL330 모터 토크 비활성화
                for dxl_id in self.dxl_ids:
                    self.packetHandler.write1ByteTxRx(
                        self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
                
                # 포트 닫기
                self.portHandler.closePort()
                
            print("XL330 cleanup complete")
            
        except Exception as e:
            print(f"XL330 cleanup error: {e}")


# XL330 전용 사용 예제
if __name__ == "__main__":
    # XL330 테스트
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    
    try:
        controller = DynamixelController(
            shm_manager=shm_manager,
            device_name='/dev/ttyUSB0',
            baudrate=57600,
            dxl_ids=[10, 11],
            current_unit_to_ma=1.0,  # XL330 전용
            max_current_limit_ma=300.0  # XL330 안전 제한
        )
        
        controller.start()
        if controller.start_wait():
            print("XL330 Controller started successfully")
            
            # XL330 테스트
            for i in range(10):
                state = controller.get_state()
                print(f"Pos: {state['gripper_position_mm']:.2f}mm, "
                      f"Current: [{state['motor_currents_ma'][0]:+.1f}, {state['motor_currents_ma'][1]:+.1f}]mA, "
                      f"Max Load: {state['max_load_current_ma']:.1f}mA, "
                      f"Total Load: {state['total_load_current_ma']:.1f}mA, "
                      f"Direction OK: {state['direction_consistent']}")
                time.sleep(1)
        else:
            print("XL330 Controller failed to start")
    
    finally:
        controller.stop()
        controller.join()
        shm_manager.shutdown()