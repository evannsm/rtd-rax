IMAGE_NAME=ws_rtd_jazzy
DOCKERHUB_IMAGE=evannsmc/rtd-rax:latest
CONTAINER_NAME=ws_rtd
PROJECT_ROOT:=$(abspath ..)
PYTHON ?= /opt/rtd-venv/bin/python
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
DOCKER_USER := $(HOST_UID):$(HOST_GID)

build:
	docker build -f Dockerfile .. -t $(IMAGE_NAME)

run:
	docker run -itd --rm --ipc host --net host --pid host \
		--user $(DOCKER_USER) \
		-v $(PROJECT_ROOT):/workspace/ws_RTD \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME)

run_gui:
	docker run -itd --rm --ipc host --net host --pid host \
		--user $(DOCKER_USER) \
		-e DISPLAY=$(DISPLAY) \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v $(PROJECT_ROOT):/workspace/ws_RTD \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME)

start:
	docker start $(CONTAINER_NAME)

stop:
	docker stop $(CONTAINER_NAME)

kill:
	docker kill $(CONTAINER_NAME)

attach:
	docker exec -it --user $(DOCKER_USER) $(CONTAINER_NAME) bash

pull:
	docker pull $(DOCKERHUB_IMAGE)
	docker tag $(DOCKERHUB_IMAGE) $(IMAGE_NAME)

push:
	docker tag $(IMAGE_NAME) $(DOCKERHUB_IMAGE)
	docker push $(DOCKERHUB_IMAGE)

rtd-numpy:
	docker exec -it --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && $(PYTHON) one_shot_rtd.py"

rtd-jax:
	docker exec -it --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_jax && $(PYTHON) one_shot_rtd.py"

FRS ?= standard
UNCERTAINTY ?= 0.01
DISTURBANCE ?= 0.0
JOURNEY_MAX_STEPS ?= 60
JOURNEY_GOAL_TOL ?= 0.10
JOURNEY_T_MOVE ?= 0.5
JOURNEY_VERIFY ?= 0
JOURNEY_VERIFY_EVERY ?= 1
JOURNEY_VERIFY_UNCERTAINTY ?= 0.01
JOURNEY_VERIFY_DISTURBANCE ?= 0.0
JOURNEY_REPAIR ?= 0
JOURNEY_REPAIR_MAX_ITERS ?= 4
JOURNEY_REPAIR_SPEED_BACKOFF ?= 0.15
JOURNEY_REPAIR_BUFFER_STEP ?= 0.01
JOURNEY_REPAIR_PUSH_SAFE ?= 0
JOURNEY_REPAIR_PUSH_ITERS ?= 2
JOURNEY_REPAIR_PUSH_BACKOFF ?= 0.03
JOURNEY_REPAIR_PUSH_K1 ?= 0
JOURNEY_REPAIR_PUSH_K1_STEP ?= 0.04
JOURNEY_REPAIR_PUSH_K1_DIR ?= auto
CASE_STUDY_SEED ?= 10
DISTURBANCE_COMPARE_SEED ?= 32
DISTURBANCE_OBSTACLE_INSET ?= 0.30
CASE_REPRESENTATIVE_SEED ?= 12
MANUSCRIPT_DISTURBANCE_SEED ?= 32
MANUSCRIPT_DISTURBANCE_TIMING_RUNS ?= 20
MANUSCRIPT_DISTURBANCE_TIMING_WARMUP ?= 12
CASE_STUDY_OUT_DIR ?= /workspace/ws_RTD/case_study_outputs
MANUSCRIPT_IMG_DIR ?= /workspace/ws_RTD/figures
MANUSCRIPT_GAP_DIR = $(MANUSCRIPT_IMG_DIR)/Study1_Gap
MANUSCRIPT_ANGLED_DIR = $(MANUSCRIPT_IMG_DIR)/Study2_AngledObs
MANUSCRIPT_DISTURBANCE_DIR = $(MANUSCRIPT_IMG_DIR)/Study3_Disturbance
# Case 2 representative run:
#   Standard RTD: goal_reached in 12 steps
#   Non-Inflated RTD: terminated in 26 steps
#   RTD-RAX: goal_reached in 12 steps with 1 repair
ANGLED_CASE2_ARGS = --v0 0.75 --x-des 1.1261383717579623 --y-des -1.95 \
	--obs-x 0.3028769328416644 --obs-y -1.2943691173614533 \
	--obs-half-width 0.14983429713340735 --obs-half-height 0.27546211625061284 \
	--max-steps 26 --goal-tol 0.10 --speed-tol 0.08 --t-move 0.45 \
	--verify-uncertainty 0.03 --verify-dt 0.01 \
	--repair-max-iters 6 --repair-speed-backoff 0.0 --repair-buffer-step 0.015 \
	--repair-push-iters 2 --repair-push-k1-step 0.5 \
	--verify-disturbance-components 0.017 0.0595 0.0425 0.017 \
	--baseline-obstacle-buffer 0.08 --rax-obstacle-buffer 0.04 \
	--execution-disturbance-components 0.02 0.1 0.06 0.03 \
	--execution-disturbance-mode episode --execution-seed 275
# Additional one-panel repair-view targets track the same representative Case 2
# run so the three-panel and one-panel outputs stay visually consistent.
ANGLED_CASE2_REPAIR_VIEW_ARGS = $(ANGLED_CASE2_ARGS)
# Case 2 alternate two-repair run:
#   Standard RTD: goal_reached in 11 steps
#   Non-Inflated RTD: terminated in 26 steps
#   RTD-RAX: goal_reached in 11 steps with 2 repairs
ANGLED_CASE2_TWO_REPAIR_ARGS = --v0 0.75 --x-des 1.1261383717579623 --y-des -1.95 \
	--obs-x 0.3028769328416644 --obs-y -1.2943691173614533 \
	--obs-half-width 0.14983429713340735 --obs-half-height 0.27546211625061284 \
	--max-steps 26 --goal-tol 0.10 --speed-tol 0.08 --t-move 0.45 \
	--verify-uncertainty 0.03 --verify-dt 0.01 \
	--repair-max-iters 2 --repair-speed-backoff 0.08 --repair-buffer-step 0.01 \
	--repair-push-iters 0 --repair-push-k1-step 0.0 \
	--verify-disturbance-components 0.017 0.0595 0.0425 0.017 \
	--baseline-obstacle-buffer 0.08 --rax-obstacle-buffer 0.04 \
	--execution-disturbance-components 0.02 0.1 0.06 0.03 \
	--execution-disturbance-mode episode --execution-seed 241

rtd-gap:
	docker exec -it --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && $(PYTHON) one_shot_rtd_gap.py --frs $(FRS)"

rtd-gap-verify:
	docker exec -it --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && $(PYTHON) one_shot_rtd_gap.py --frs $(FRS) --verify --uncertainty $(UNCERTAINTY) --disturbance $(DISTURBANCE)"

rtd-journey-gap:
	docker exec -it --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && $(PYTHON) rtd_gap_journey.py --frs $(FRS) --max-steps $(JOURNEY_MAX_STEPS) --goal-tol $(JOURNEY_GOAL_TOL) --t-move $(JOURNEY_T_MOVE) --verify-every $(JOURNEY_VERIFY_EVERY) --verify-uncertainty $(JOURNEY_VERIFY_UNCERTAINTY) --verify-disturbance $(JOURNEY_VERIFY_DISTURBANCE) $(if $(filter 1,$(JOURNEY_VERIFY)),--verify,)"

rtd-journey-gap-compare:
	docker exec -it --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && $(PYTHON) rtd_gap_journey_compare.py --max-steps $(JOURNEY_MAX_STEPS) --goal-tol $(JOURNEY_GOAL_TOL) --t-move $(JOURNEY_T_MOVE) --verify-every $(JOURNEY_VERIFY_EVERY) --verify-uncertainty $(JOURNEY_VERIFY_UNCERTAINTY) --verify-disturbance $(JOURNEY_VERIFY_DISTURBANCE) --repair-max-iters $(JOURNEY_REPAIR_MAX_ITERS) --repair-speed-backoff $(JOURNEY_REPAIR_SPEED_BACKOFF) --repair-buffer-step $(JOURNEY_REPAIR_BUFFER_STEP) --repair-push-iters $(JOURNEY_REPAIR_PUSH_ITERS) --repair-push-backoff $(JOURNEY_REPAIR_PUSH_BACKOFF) --repair-push-k1-step $(JOURNEY_REPAIR_PUSH_K1_STEP) --repair-push-k1-dir $(JOURNEY_REPAIR_PUSH_K1_DIR) $(if $(filter 1,$(JOURNEY_VERIFY)),--verify,) $(if $(filter 1,$(JOURNEY_REPAIR)),--repair-on-immrax,) $(if $(filter 1,$(JOURNEY_REPAIR_PUSH_SAFE)),--repair-push-safe,) $(if $(filter 1,$(JOURNEY_REPAIR_PUSH_K1)),--repair-push-k1,)"

rtd-journey-angled-compare:
	docker exec -it --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && $(PYTHON) rtd_angled_obstacle_compare.py --max-steps $(JOURNEY_MAX_STEPS) --goal-tol $(JOURNEY_GOAL_TOL) --t-move $(JOURNEY_T_MOVE) --verify-every $(JOURNEY_VERIFY_EVERY) --verify-uncertainty $(JOURNEY_VERIFY_UNCERTAINTY) --verify-disturbance $(JOURNEY_VERIFY_DISTURBANCE) --repair-max-iters $(JOURNEY_REPAIR_MAX_ITERS) --repair-speed-backoff $(JOURNEY_REPAIR_SPEED_BACKOFF) --repair-buffer-step $(JOURNEY_REPAIR_BUFFER_STEP) --repair-push-iters $(JOURNEY_REPAIR_PUSH_ITERS) --repair-push-backoff $(JOURNEY_REPAIR_PUSH_BACKOFF) --repair-push-k1-step $(JOURNEY_REPAIR_PUSH_K1_STEP) --repair-push-k1-dir $(JOURNEY_REPAIR_PUSH_K1_DIR) $(if $(filter 1,$(JOURNEY_VERIFY)),--verify,) $(if $(filter 1,$(JOURNEY_REPAIR)),--repair-on-immrax,) $(if $(filter 1,$(JOURNEY_REPAIR_PUSH_SAFE)),--repair-push-safe,) $(if $(filter 1,$(JOURNEY_REPAIR_PUSH_K1)),--repair-push-k1,)"

rtd-disturbance-compare:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(CASE_STUDY_OUT_DIR) && $(PYTHON) rtd_random_disturbance_compare.py --seed $(DISTURBANCE_COMPARE_SEED) --obstacle-inset $(DISTURBANCE_OBSTACLE_INSET) --verify-uncertainty 0.01 --repair-max-iters 4 --repair-speed-backoff 0.08 --repair-buffer-step 0.015 --save-fig $(CASE_STUDY_OUT_DIR)/disturbance_compare_seed$(DISTURBANCE_COMPARE_SEED).pdf --save-animation $(CASE_STUDY_OUT_DIR)/disturbance_compare_seed$(DISTURBANCE_COMPARE_SEED).gif --no-show"

manuscript-disturbance-compare:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_DISTURBANCE_DIR) && $(PYTHON) rtd_random_disturbance_compare.py --seed $(MANUSCRIPT_DISTURBANCE_SEED) --obstacle-inset $(DISTURBANCE_OBSTACLE_INSET) --verify-uncertainty 0.01 --verify-dt 0.01 --repair-max-iters 4 --repair-speed-backoff 0.08 --repair-buffer-step 0.015 --save-fig $(MANUSCRIPT_DISTURBANCE_DIR)/disturbance_compare_seed$(MANUSCRIPT_DISTURBANCE_SEED).pdf --save-animation $(MANUSCRIPT_DISTURBANCE_DIR)/disturbance_compare_seed$(MANUSCRIPT_DISTURBANCE_SEED).gif --no-show"

manuscript-disturbance-seed10:
	$(MAKE) manuscript-disturbance-compare MANUSCRIPT_DISTURBANCE_SEED=10 DISTURBANCE_OBSTACLE_INSET=0.0

manuscript-disturbance-seed20:
	$(MAKE) manuscript-disturbance-compare MANUSCRIPT_DISTURBANCE_SEED=20 DISTURBANCE_OBSTACLE_INSET=0.0

manuscript-disturbance-seed32:
	$(MAKE) manuscript-disturbance-compare MANUSCRIPT_DISTURBANCE_SEED=32 DISTURBANCE_OBSTACLE_INSET=0.30

manuscript-disturbance-gallery:
	$(MAKE) manuscript-disturbance-seed10
	$(MAKE) manuscript-disturbance-seed20
	$(MAKE) manuscript-disturbance-seed32

manuscript-disturbance-timing:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_DISTURBANCE_DIR) && $(PYTHON) rtd_case3_timing_benchmark.py --seed $(MANUSCRIPT_DISTURBANCE_SEED) --runs $(MANUSCRIPT_DISTURBANCE_TIMING_RUNS) --warmup-runs $(MANUSCRIPT_DISTURBANCE_TIMING_WARMUP) --obstacle-inset $(DISTURBANCE_OBSTACLE_INSET) --verify-uncertainty 0.01 --verify-dt 0.01 --repair-max-iters 4 --repair-speed-backoff 0.08 --repair-buffer-step 0.015 --save-csv $(MANUSCRIPT_DISTURBANCE_DIR)/disturbance_timing_seed$(MANUSCRIPT_DISTURBANCE_SEED)_n$(MANUSCRIPT_DISTURBANCE_TIMING_RUNS).csv --save-summary-json $(MANUSCRIPT_DISTURBANCE_DIR)/disturbance_timing_seed$(MANUSCRIPT_DISTURBANCE_SEED)_n$(MANUSCRIPT_DISTURBANCE_TIMING_RUNS).json --save-table-tex $(MANUSCRIPT_DISTURBANCE_DIR)/disturbance_timing_seed$(MANUSCRIPT_DISTURBANCE_SEED)_n$(MANUSCRIPT_DISTURBANCE_TIMING_RUNS)_table.tex"

manuscript-gap-noerror-verify:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --world-legend none --hide-result-text --hide-verify-text --hide-v0-in-title --custom-title 'RTD-RAX | Safe' --save-world-fig $(MANUSCRIPT_GAP_DIR)/gap_noerror_verify.pdf --no-show"

manuscript-gap-noerror-verify-legend:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --world-legend outside --hide-result-text --hide-verify-text --custom-title 'RTD-RAX | Safe' --hide-v0-in-title --world-title 'World Frame' --save-world-fig $(MANUSCRIPT_GAP_DIR)/gap_noerror_verify_legend.pdf --no-show"

manuscript-gap-standard-fail:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs standard --world-legend none --hide-result-text --hide-verify-text --hide-v0-in-title --custom-title 'RTD Standard | Infeasible' --save-world-fig $(MANUSCRIPT_GAP_DIR)/gap_standard_fail.pdf --no-show"

manuscript-gap-standard-fail-legend:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs standard --world-legend outside --hide-result-text --hide-verify-text --custom-title 'RTD Standard | Infeasible' --hide-v0-in-title --world-title 'World Frame' --save-world-fig $(MANUSCRIPT_GAP_DIR)/gap_standard_fail_legend.pdf --no-show"

manuscript-full-gap-noerror-verify-legend:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --panel-world-legend outside_top --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --hide-immrax-nominal --hide-start-marker --hide-current-footprint-arrow --custom-title '' --world-title 'World Frame' --save-full-fig $(MANUSCRIPT_GAP_DIR)/full_gap_noerror_verify_legend.pdf --no-show"

manuscript-full-gap-noerror-verify-no-legend:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --panel-world-legend none --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --hide-immrax-nominal --hide-start-marker --hide-current-footprint-arrow --custom-title '' --world-title 'World Frame' --save-full-fig $(MANUSCRIPT_GAP_DIR)/full_gap_noerror_verify_no_legend.pdf --no-show"

manuscript-full-gap-noerror-verify-compact-legend:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --compact-full-fig --panel-world-legend outside_top --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --hide-immrax-nominal --hide-start-marker --hide-current-footprint-arrow --show-goal --custom-title '' --world-title 'World Frame' --save-full-fig $(MANUSCRIPT_GAP_DIR)/full_gap_noerror_verify_compact_legend.pdf --no-show"

manuscript-full-gap-noerror-verify-compact-no-legend:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --compact-full-fig --panel-world-legend none --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --hide-immrax-nominal --hide-start-marker --hide-current-footprint-arrow --show-goal --custom-title '' --world-title 'World Frame' --save-full-fig $(MANUSCRIPT_GAP_DIR)/full_gap_noerror_verify_compact_no_legend.pdf --no-show"

manuscript-full-gap-noerror-verify-compact-legend-horiz:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --compact-full-fig --compact-full-layout horizontal --panel-world-legend outside_top --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --hide-immrax-nominal --hide-start-marker --hide-current-footprint-arrow --show-goal --custom-title '' --world-title 'World Frame' --save-full-fig $(MANUSCRIPT_GAP_DIR)/full_gap_noerror_verify_compact_legend_horiz.pdf --no-show"

manuscript-full-gap-noerror-verify-compact-no-legend-horiz:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --compact-full-fig --compact-full-layout horizontal --panel-world-legend none --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --hide-immrax-nominal --hide-start-marker --hide-current-footprint-arrow --show-goal --custom-title '' --world-title 'World Frame' --save-full-fig $(MANUSCRIPT_GAP_DIR)/full_gap_noerror_verify_compact_no_legend_horiz.pdf --no-show"

manuscript-full-gap-noerror-verify-compact-legend-horiz-right:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --compact-full-fig --compact-full-layout horizontal --compact-full-legend-placement right --panel-world-legend outside_top --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --hide-immrax-nominal --hide-start-marker --hide-current-footprint-arrow --show-goal --custom-title '' --world-title 'World Frame' --save-full-fig $(MANUSCRIPT_GAP_DIR)/full_gap_noerror_verify_compact_legend_horiz_right.pdf --no-show"

manuscript-full-gap-noerror-verify-compact-legend-horiz-left:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --compact-full-fig --compact-full-layout horizontal --compact-full-legend-placement left --panel-world-legend outside_top --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --hide-immrax-nominal --hide-start-marker --hide-current-footprint-arrow --show-goal --custom-title '' --world-title 'World Frame' --save-full-fig $(MANUSCRIPT_GAP_DIR)/full_gap_noerror_verify_compact_legend_horiz_left.pdf --no-show"

manuscript-angled-compare-legend-left:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_ANGLED_DIR) && $(PYTHON) rtd_angled_obstacle_compare.py --verify --verify-every 1 --verify-uncertainty 0.03 --verify-disturbance 0.03 --repair-on-immrax --repair-max-iters 2 --repair-speed-backoff 0.08 --repair-buffer-step 0.01 --repair-push-safe --repair-push-iters 2 --repair-push-backoff 0.03 --repair-push-k1 --repair-push-k1-step 0.29 --repair-push-k1-dir auto --world-legend outside_left_top --save-full-fig $(MANUSCRIPT_ANGLED_DIR)/angled_compare_legend_left.pdf --no-show"

manuscript-angled-compare-no-legend:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_ANGLED_DIR) && $(PYTHON) rtd_angled_obstacle_compare.py --verify --verify-every 1 --verify-uncertainty 0.03 --verify-disturbance 0.03 --repair-on-immrax --repair-max-iters 2 --repair-speed-backoff 0.08 --repair-buffer-step 0.01 --repair-push-safe --repair-push-iters 2 --repair-push-backoff 0.03 --repair-push-k1 --repair-push-k1-step 0.29 --repair-push-k1-dir auto --world-legend none --save-full-fig $(MANUSCRIPT_ANGLED_DIR)/angled_compare_no_legend.pdf --no-show"

manuscript-merge-gap:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs standard --overlay-noerror-frs --show-start-footprint --hide-current-footprint --footprint-color-mode compare --show-goal --world-legend none --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --custom-title 'RTD Standard | Infeasible' --font-bump 4 --suptitle-bump 1 --save-world-fig $(MANUSCRIPT_GAP_DIR)/gap_standard_fail_clean.pdf --no-show"
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --hide-frs-contour --hide-immrax-nominal --show-start-footprint --hide-current-footprint --footprint-color-mode compare --show-goal --world-legend none --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --custom-title 'RTD-RAX | Feasible & Safe' --font-bump 4 --suptitle-bump 2 --save-world-fig $(MANUSCRIPT_GAP_DIR)/gap_noerror_verify_clean.pdf --no-show"
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "$(PYTHON) -c \"from pypdf import PdfReader, PdfWriter, PageObject, Transformation; fig_dir='$(MANUSCRIPT_GAP_DIR)'; left=PdfReader(f'{fig_dir}/gap_standard_fail_clean.pdf'); right=PdfReader(f'{fig_dir}/gap_noerror_verify_clean.pdf'); lp=left.pages[0]; rp=right.pages[0]; lw=float(lp.mediabox.width); lh=float(lp.mediabox.height); rw=float(rp.mediabox.width); rh=float(rp.mediabox.height); s=lh/rh; rp.add_transformation(Transformation().scale(s,s)); rw2=rw*s; rh2=rh*s; gap=10; tw=lw+rw2+gap; th=lh; m=PageObject.create_blank_page(width=tw,height=th); m.merge_translated_page(lp,0,0); m.merge_translated_page(rp,lw+gap,(th-rh2)/2); w=PdfWriter(); w.add_page(m); f=open(f'{fig_dir}/merge_gap_clean_small.pdf','wb'); w.write(f); f.close(); print('Saved merge_gap_clean_small.pdf')\""

manuscript-merge-gap-legend:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) one_shot_rtd_gap.py --frs standard --overlay-noerror-frs --show-start-footprint --hide-current-footprint --footprint-color-mode compare --show-goal --world-legend none --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --custom-title 'RTD Standard | Infeasible' --font-bump 4 --suptitle-bump 1 --save-world-fig $(MANUSCRIPT_GAP_DIR)/gap_standard_fail_legend_clean.pdf --no-show"
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && $(PYTHON) one_shot_rtd_gap.py --frs noerror --verify --uncertainty 0.01 --disturbance 0.0 --hide-frs-contour --hide-immrax-nominal --show-start-footprint --hide-current-footprint --footprint-color-mode compare --show-goal --world-legend outside --hide-result-text --hide-verify-text --hide-v0-in-title --hide-gap-annotation --custom-title 'RTD-RAX | Feasible & Safe' --font-bump 4 --suptitle-bump 2 --legend-include-nominal-frs --save-world-fig $(MANUSCRIPT_GAP_DIR)/gap_noerror_verify_legend_clean.pdf --no-show"
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "$(PYTHON) -c \"from pypdf import PdfReader, PdfWriter, PageObject, Transformation; fig_dir='$(MANUSCRIPT_GAP_DIR)'; left=PdfReader(f'{fig_dir}/gap_standard_fail_legend_clean.pdf'); right=PdfReader(f'{fig_dir}/gap_noerror_verify_legend_clean.pdf'); lp=left.pages[0]; rp=right.pages[0]; lw=float(lp.mediabox.width); lh=float(lp.mediabox.height); rw=float(rp.mediabox.width); rh=float(rp.mediabox.height); s=lh/rh; rp.add_transformation(Transformation().scale(s,s)); rw2=rw*s; rh2=rh*s; gap=10; tw=lw+rw2+gap; th=lh; m=PageObject.create_blank_page(width=tw,height=th); m.merge_translated_page(lp,0,0); m.merge_translated_page(rp,lw+gap,(th-rh2)/2); w=PdfWriter(); w.add_page(m); f=open(f'{fig_dir}/merge_gap_clean_small_legend.pdf','wb'); w.write(f); f.close(); print('Saved merge_gap_clean_small_legend.pdf')\""

# Case 1: Gap animation (Standard RTD vs RTD-RAX)
rtd-gap-animate:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(CASE_STUDY_OUT_DIR) && $(PYTHON) rtd_gap_animate.py --verify-uncertainty 0.01 --save-fig $(CASE_STUDY_OUT_DIR)/gap_compare.pdf --save-animation $(CASE_STUDY_OUT_DIR)/gap_compare.gif --no-show"

manuscript-gap-animate:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_GAP_DIR) && $(PYTHON) rtd_gap_animate.py --verify-uncertainty 0.01 --save-animation $(MANUSCRIPT_GAP_DIR)/gap_compare.gif --no-show"
	$(MAKE) manuscript-merge-gap
	$(MAKE) manuscript-merge-gap-legend

# Case 2: Angled obstacle animation (Standard vs Non-Inflated vs RTD-RAX)
rtd-angled-animate:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(CASE_STUDY_OUT_DIR) && $(PYTHON) rtd_angled_obstacle_animate.py $(ANGLED_CASE2_ARGS) --save-fig $(CASE_STUDY_OUT_DIR)/angled_animate.pdf --save-animation $(CASE_STUDY_OUT_DIR)/angled_animate.gif --save-summary-json $(CASE_STUDY_OUT_DIR)/angled_animate_path_lengths.json --no-show"

manuscript-angled-animate:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_ANGLED_DIR) && $(PYTHON) rtd_angled_obstacle_animate.py $(ANGLED_CASE2_ARGS) --save-fig $(MANUSCRIPT_ANGLED_DIR)/angled_animate_legend.pdf --save-fig-no-legend $(MANUSCRIPT_ANGLED_DIR)/angled_animate_no_legend.pdf --save-animation $(MANUSCRIPT_ANGLED_DIR)/angled_animate.gif --save-summary-json $(MANUSCRIPT_ANGLED_DIR)/angled_animate_path_lengths.json --no-show"

# Case 2: Additional one-panel RTD-RAX repair view
rtd-angled-repair-view:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(CASE_STUDY_OUT_DIR) && $(PYTHON) rtd_angled_obstacle_animate.py $(ANGLED_CASE2_REPAIR_VIEW_ARGS) --save-repair-view-fig $(CASE_STUDY_OUT_DIR)/angled_repair_view.pdf --save-repair-view-animation $(CASE_STUDY_OUT_DIR)/angled_repair_view.gif --no-show"

manuscript-angled-repair-view:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_ANGLED_DIR) && $(PYTHON) rtd_angled_obstacle_animate.py $(ANGLED_CASE2_REPAIR_VIEW_ARGS) --save-repair-view-fig $(MANUSCRIPT_ANGLED_DIR)/angled_repair_view_legend.pdf --save-repair-view-fig-no-legend $(MANUSCRIPT_ANGLED_DIR)/angled_repair_view_no_legend.pdf --save-repair-view-animation $(MANUSCRIPT_ANGLED_DIR)/angled_repair_view.gif --no-show"

# Case 2: Alternate two-repair comparison and repair-view outputs
rtd-angled-animate-two-repairs:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(CASE_STUDY_OUT_DIR) && $(PYTHON) rtd_angled_obstacle_animate.py $(ANGLED_CASE2_TWO_REPAIR_ARGS) --save-fig $(CASE_STUDY_OUT_DIR)/angled_animate_two_repairs.pdf --save-animation $(CASE_STUDY_OUT_DIR)/angled_animate_two_repairs.gif --save-summary-json $(CASE_STUDY_OUT_DIR)/angled_animate_two_repairs_path_lengths.json --no-show"

manuscript-angled-animate-two-repairs:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_ANGLED_DIR) && $(PYTHON) rtd_angled_obstacle_animate.py $(ANGLED_CASE2_TWO_REPAIR_ARGS) --save-fig $(MANUSCRIPT_ANGLED_DIR)/angled_animate_two_repairs_legend.pdf --save-fig-no-legend $(MANUSCRIPT_ANGLED_DIR)/angled_animate_two_repairs_no_legend.pdf --save-animation $(MANUSCRIPT_ANGLED_DIR)/angled_animate_two_repairs.gif --save-summary-json $(MANUSCRIPT_ANGLED_DIR)/angled_animate_two_repairs_path_lengths.json --no-show"

rtd-angled-repair-view-two-repairs:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(CASE_STUDY_OUT_DIR) && $(PYTHON) rtd_angled_obstacle_animate.py $(ANGLED_CASE2_TWO_REPAIR_ARGS) --save-repair-view-fig $(CASE_STUDY_OUT_DIR)/angled_repair_view_two_repairs.pdf --save-repair-view-animation $(CASE_STUDY_OUT_DIR)/angled_repair_view_two_repairs.gif --no-show"

manuscript-angled-repair-view-two-repairs:
	docker exec --user $(DOCKER_USER) $(CONTAINER_NAME) bash -lc "cd /workspace/ws_RTD/turtlebot_rtd_numpy && mkdir -p $(MANUSCRIPT_ANGLED_DIR) && $(PYTHON) rtd_angled_obstacle_animate.py $(ANGLED_CASE2_TWO_REPAIR_ARGS) --save-repair-view-fig $(MANUSCRIPT_ANGLED_DIR)/angled_repair_view_two_repairs_legend.pdf --save-repair-view-fig-no-legend $(MANUSCRIPT_ANGLED_DIR)/angled_repair_view_two_repairs_no_legend.pdf --save-repair-view-animation $(MANUSCRIPT_ANGLED_DIR)/angled_repair_view_two_repairs.gif --no-show"

rtd-case2-suite:
	$(MAKE) rtd-angled-animate
	$(MAKE) rtd-angled-repair-view
	$(MAKE) rtd-angled-animate-two-repairs
	$(MAKE) rtd-angled-repair-view-two-repairs

manuscript-case1-gap-suite:
	$(MAKE) manuscript-gap-standard-fail
	$(MAKE) manuscript-gap-standard-fail-legend
	$(MAKE) manuscript-gap-noerror-verify
	$(MAKE) manuscript-gap-noerror-verify-legend
	$(MAKE) manuscript-full-gap-noerror-verify-legend
	$(MAKE) manuscript-full-gap-noerror-verify-no-legend
	$(MAKE) manuscript-full-gap-noerror-verify-compact-legend
	$(MAKE) manuscript-full-gap-noerror-verify-compact-no-legend
	$(MAKE) manuscript-full-gap-noerror-verify-compact-legend-horiz
	$(MAKE) manuscript-full-gap-noerror-verify-compact-no-legend-horiz
	$(MAKE) manuscript-full-gap-noerror-verify-compact-legend-horiz-right
	$(MAKE) manuscript-full-gap-noerror-verify-compact-legend-horiz-left
	$(MAKE) manuscript-gap-animate

manuscript-case2-suite:
	$(MAKE) manuscript-angled-animate
	$(MAKE) manuscript-angled-repair-view
	$(MAKE) manuscript-angled-animate-two-repairs
	$(MAKE) manuscript-angled-repair-view-two-repairs

manuscript-figures:
	$(MAKE) manuscript-disturbance-gallery
	$(MAKE) manuscript-case1-gap-suite
	$(MAKE) manuscript-angled-compare-legend-left
	$(MAKE) manuscript-angled-compare-no-legend
	$(MAKE) manuscript-case2-suite

fix-perms:
	docker exec --user 0:0 $(CONTAINER_NAME) chown -R $(DOCKER_USER) /workspace/ws_RTD/manuscript_work
