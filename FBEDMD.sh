#!/bin/bash
for i in {1..30}
do
	mprof run --include-children --output build/stats/FBEDMD/FBEDMD.dat main.py profiler=True lifting_functions@pykoop_pipeline=soft_robot_poly2_centers10 regressors@pykoop_pipeline=FBEDMD
	mprof run --include-children --output build/stats/FBEDMD/FBEDMD25.dat main.py profiler=True lifting_functions@pykoop_pipeline=soft_robot_poly2_centers25 regressors@pykoop_pipeline=FBEDMD
	mprof run --include-children --output build/stats/FBEDMD/FBEDMD50.dat main.py profiler=True lifting_functions@pykoop_pipeline=soft_robot_poly2_centers50 regressors@pykoop_pipeline=FBEDMD
done
