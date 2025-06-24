#!/bin/bash
for i in {1..30}
do
	mprof run --include-children --output build/stats/FBEDMD-AS/FBEDMD-AS.dat main.py profiler=True lifting_functions@pykoop_pipeline=soft_robot_poly2_centers10 regressors@pykoop_pipeline=FBEDMD-AS
	mprof run --include-children --output build/stats/FBEDMD-AS/FBEDMD-AS25.dat main.py profiler=True lifting_functions@pykoop_pipeline=soft_robot_poly2_centers25 regressors@pykoop_pipeline=FBEDMD-AS
	mprof run --include-children --output build/stats/FBEDMD-AS/FBEDMD-AS50.dat main.py profiler=True lifting_functions@pykoop_pipeline=soft_robot_poly2_centers50 regressors@pykoop_pipeline=FBEDMD-AS
done
