"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace

(umi) kist@kist:~/Umi_ws/universal_manipulation_interface$ accelerate config
----------------------------------------In which compute environment are you running?
This machine                                                                                                                                                                             
----------------------------------------Which type of machine are you using?                                                                                                                                                     
multi-GPU                                                                                                                                                                                
How many different machines will you use (use more than 1 for multi-node training)? [1]:                                                                                                 
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:                                                           
Do you wish to optimize your script with torch dynamo?[yes/NO]:                                                                                                                          
Do you want to use DeepSpeed? [yes/NO]:                                                                                                                                                  
Do you want to use FullyShardedDataParallel? [yes/NO]:                                                                                                                                   
Do you want to use Megatron-LM ? [yes/NO]:                                                                                                                                               
How many GPU(s) should be used for distributed training? [1]:2
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:0,1
---------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
no                                                                                                                                                                                       
accelerate configuration saved at /home/kist/.cache/huggingface/accelerate/default_config.yaml         

accelerate launch --num_processes 2 \
train.py \
--config-name=train_diffusion_unet_timm_umi_workspace \
task.dataset_path=example_demo_session/dataset.zarr.zip \
training.resume=True \
multi_run.run_dir=/home/kist/Umi_ws/universal_manipulation_interface/data/outputs/2025-07-31/16-02-50_train_diffusion_unet_timm_umi
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
