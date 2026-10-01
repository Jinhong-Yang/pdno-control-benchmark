import numpy as np

from pdno.evaluation.closed_loop import _obs_for_tick


def test_online_observation_history_excludes_events_before_receive_time():
    sensor_capture=np.zeros((4,16),np.int16); sensor_receive=np.zeros((4,16),np.int16)
    sensor_capture[3,:]=1; sensor_receive[3,:]=3
    sensor_values=np.stack([np.arange(16),100+np.arange(16),200+np.arange(16),300+np.arange(16)]).astype(np.float32)
    image=np.ones((16,64),np.float32)
    sched={"sensor_capture":sensor_capture,"sensor_receive":sensor_receive,"image_capture":np.array([0,0,0,1]),"image_receive":np.array([0,0,0,3])}
    images=[(image,np.ones_like(image,bool)),(image*2,np.ones_like(image,bool)),None,None]
    actions=[np.zeros(2,np.float32),np.ones(2,np.float32)*0.1]
    obs,prev=_obs_for_tick(actions,2,"burgers",np.zeros(8,np.float32),np.zeros(256,np.float32),sched,sensor_values,images)
    assert obs["sensor_value"][-1,0] == 0
    assert obs["sensor_capture_time"][-1,0] == 0
    assert obs["sensor_receive_time"][-1,0] == 0
    assert obs["image_valid"] and obs["image_capture_time"] == 0
    assert obs["image_receive_time"] == 0
    assert np.all(obs["sensor_capture_time"] <= 2)
    assert np.all(obs["sensor_receive_time"] <= 2)
    assert np.allclose(prev,actions[-1])
    obs3,_=_obs_for_tick(actions+[np.ones(2,np.float32)*.2],3,"burgers",np.zeros(8,np.float32),np.zeros(256,np.float32),sched,sensor_values,images)
    assert obs3["sensor_value"][-1,0] == 100
    assert obs3["sensor_capture_time"][-1,0] == 1
    assert obs3["sensor_receive_time"][-1,0] == 3
    assert obs3["image_capture_time"] == 1 and obs3["image_receive_time"] == 3
