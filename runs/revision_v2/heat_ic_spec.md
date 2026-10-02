# Positive heat initial-condition contract

Fixed before generation: u(x) = A sin(pi x) [0.2 + v(x)^2] / max_grid {sin(pi x)[0.2+v(x)^2]}, with v(x)=sum(k=1..8) z_k sin(k pi x)/k, independent standard-normal z_k and A uniform [0.1,0.5]. Truth grid has 256 interior samples and homogeneous Dirichlet boundaries. Initial values lie in [0,0.5], below the unchanged original train/validation threshold q_max=2.1322593092918396. Original parameter, actuator, goal, noise and action definitions remain in effect under fresh revision parent identifiers.

Because the initial-condition distribution changes, retrain the observer and all six learned controller families on fresh heat train parents. Validation controls checkpoint selection. Freeze policies before generating/evaluating the new test parents. Any-time violation, violation duration, maximum magnitude, integrated mean magnitude and cost decomposition are recorded. Original signed-heat data are retained as original-campaign evidence.
