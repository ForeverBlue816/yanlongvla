"""Official LIBERO-Plus simulator, original pi0.5 observation/action handling."""
import argparse,collections,importlib.util,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from openpi_client import image_tools,websocket_client_policy
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ROOT=Path('/projects/yanlongvla')
spec=importlib.util.spec_from_file_location('upstream_libero_main',ROOT/'src/openpi/examples/libero/main.py')
upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
LIMITS=dict(libero_spatial=220,libero_object=280,libero_goal=300,libero_10=520)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--units',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--port',type=int,required=True);a=parser.parse_args()
    units=json.loads(a.units.read_text());done=set()
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            row=json.loads(line);key=(row['suite'],row['task']);assert key not in done;done.add(key)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    suites={name:benchmark.get_benchmark_dict()[name](task_order_index=0) for name in {r['suite'] for r in units}}
    for entry in units:
        if (entry['suite'],entry['task']) in done:continue
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        suite=suites[entry['suite']];task=suite.get_task(entry['task']);assert task.name==entry['name']
        env=None;client=None;t=calls=0;success=False;start=time.perf_counter()
        noise_seed=int(np.random.SeedSequence([7,SUITES.index(entry['suite']),entry['task'],0]).generate_state(1)[0])
        try:
            # Official Plus parses viewpoint/robot parameters from the string.
            bddl=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file)
            env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=upstream.LIBERO_ENV_RESOLUTION,camera_widths=upstream.LIBERO_ENV_RESOLUTION)
            env.seed(7);states=suite.get_task_init_states(entry['task']);assert len(states)>=1
            env.reset();obs=env.set_init_state(states[0]);plan=collections.deque()
            while t<LIMITS[entry['suite']]+10:
                if t<10:obs,_,_,_=env.step(upstream.LIBERO_DUMMY_ACTION);t+=1;continue
                if not plan:
                    image=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
                    element={'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(image,224,224)),
                        'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist,224,224)),
                        'observation/state':np.concatenate((obs['robot0_eef_pos'],upstream._quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos'])),
                        'prompt':str(task.language)}
                    if calls==0:element['_vla_episode_seed']=noise_seed
                    if client is None:client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',a.port)
                    actions=client.infer(element)['actions'];assert actions.shape==(10,7) and np.isfinite(actions).all()
                    calls+=1;plan.extend(actions[:5])
                obs,_,success,_=env.step(plan.popleft().tolist());t+=1
                if success:break
            row=dict(entry,noise_seed=noise_seed,success=bool(success),policy_calls=calls,steps_including_settle=t,seconds=time.perf_counter()-start)
            with a.output.open('a') as handle:handle.write(json.dumps(row)+'\n');handle.flush();os.fsync(handle.fileno())
            print(json.dumps(row),flush=True)
        except Exception as exc:
            with a.output.with_suffix('.errors.jsonl').open('a') as handle:handle.write(json.dumps(dict(entry,error=repr(exc)))+'\n')
            raise
        finally:
            if client is not None:client._ws.close()
            if env is not None:env.close()


if __name__=='__main__':main()
