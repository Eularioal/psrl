# 辅助脚本

[上级导航](../README.md) · [总导航](../../../README.md)

配套正文说明研究问题、运行条件和证据边界：

- [按问题找实验报告](../../experiments/README.md)
- [查机制与 CPU 教学示例](../../knowledge/README.md)

下方列出脚本与产物。运行含 GPU、容器或训练的脚本前，按正文限定资源与时间预算。

<!-- figure:evidence-levels -->
这张图沿同一个场景解释：一次检查通过，究竟能支持多大的结论？

[![验证证据：不同实验分别回答不同问题（1/2）](../../../_static/img/document-figures/evidence-levels-screen.png)](../../../_static/img/document-figures/evidence-levels.png)

[![验证证据：不同实验分别回答不同问题（2/2）](../../../_static/img/document-figures/evidence-levels-screen-02.png)](../../../_static/img/document-figures/evidence-levels.png)

同一套调度代码可能通过 CPU 回归，但容器 CUDA、真实评分和训练仍各有条件。 验证是结论与证据的对应关系，不是每次必须爬完的一整套阶梯。 证据类别示意，不把进程存活、历史成功或测试替身写成本次完整训练。 [查看矢量图](../../../_static/img/document-figures/evidence-levels.svg)。
<!-- /figure:evidence-levels -->

## 文件与子目录

- [audit_gpu_control_startup.py](audit_gpu_control_startup.py) 读取视觉两步的历史启动事件，保存日志证据并生成初始化分段和重叠准备图。执行`python -I docs/env-worker-deep-dive/env-service-verification/scripts/audit_gpu_control_startup.py`，已有证据可离线重绘，不启动训练。
- [probe_gpu_control_startup.py](probe_gpu_control_startup.py) 在独立Ray中检验真实服务类句柄的导入成本与准备阶段的准入阻塞。需项目依赖、`CUDA_VISIBLE_DEVICES=""`、两颗CPU、零GPU、外层480秒时限及独立`--output`。其中权重准备使用30秒教学等待，不能当作真实PSRL提速。

- [verify_container_lookup.py](verify_container_lookup.py)：在本机 Docker 创建一个无 GPU 的有界容器，验证直接 API 与 Compose ID 一致、重建与停止后的身份变化，并采集自建查询的 strace。需要项目环境、已有镜像与 strace；参数、预算和结果见[查询修复](../../knowledge/01-架构与执行/06-工程审查_待修复问题与优先级.md#容器查询修复与-trace-能回答的问题)。
- [draw_container_lookup.py](draw_container_lookup.py)：从 `container-query-20261126/result.json` 绘制查询耗时与 trace 边界。在仓库根目录运行 `python docs/env-worker-deep-dive/env-service-verification/scripts/draw_container_lookup.py`，生成 experiments 下的 `container-lookup-trace.png` 与 SVG；需要 Matplotlib 和 DroidSansFallback 字体，不启动实验。

- [plot_reservation_dependency.py](plot_reservation_dependency.py) 绘制[自适应持卡研究切口](../../milestones/05-调度研究_研究全景与证据详解.md#论文还缺什么自适应持卡的研究切口)中的固定两批教学时间线。在仓库根目录运行 `python -I docs/env-worker-deep-dive/env-service-verification/scripts/plot_reservation_dependency.py`，生成 experiments 下的 `reservation-batch-dependency.png` 与 SVG；需要 Matplotlib 和 DroidSansFallback 字体，不使用 GPU，也不把教学数字作为实测结果。

- 新增实测图从已核验的比较 JSON 读取数据。在仓库根目录运行 python -I docs/env-worker-deep-dive/env-service-verification/scripts/plot_partition_latest.py，生成十步、多场景及四场景相对用时合并图的 PNG 和 SVG。合并图以各场景自己的 Command 静态为100，同时显示训练与完整时间。十步 Command 静态采用独立补跑，原失败记录保留。 [绘图脚本](plot_partition_latest.py) · [研究报告](../../experiments/12-研究报告_PSRL环境GPU调度的设计与评估.md)

- [build_env_scheduling_report.py](build_env_scheduling_report.py) 从[研究报告正文](../../experiments/12-研究报告_PSRL环境GPU调度的设计与评估.md)和现有彩图生成 A4 PDF。使用安装了 Markdown、WeasyPrint 和中文字体的文档环境，在仓库根目录运行 `python -I docs/env-worker-deep-dive/env-service-verification/scripts/build_env_scheduling_report.py`。可用 `--font` 指定中文 TrueType 字体。生成文件位于 experiments 目录。
- 八卡对照默认每组运行三个策略各 2 个 PSRL step，长档显式传入 `--steps 10` 并采用同档预算。`--arms` 用于单组补测。每组收尾后写入 `comparison.json`，包含逐步用时、累计时间和设备策略核验。 [expanded_partition_batch.py](expanded_partition_batch.py) · [八卡两档与单组补测协议](../../experiments/13-扩展对照_八卡单批次与多负载回放.md#单批次补测与十步对照)
- [replay_partition_batches.py](replay_partition_batches.py) 用已完成的轨迹日志回放不同卡数、推理服务时间与工具耗时。每个随机种子向三种策略提供相同的阶段耗时。在项目环境运行 `--help` 查看三个实测输入入口和 JSON 输出路径。
- [plot_partition_diversity.py](plot_partition_diversity.py) 从 `replay_partition_batches.py --suite diverse` 的 JSON 绘制 Episode 静态、Command 静态和动态组的耗时对照。
- [partition_vision_worker.py](partition_vision_worker.py) 由八卡实验启动器冻结进 MLGym 场景，复用分类与图像描述任务的 CUDA 训练和独立评分。
- [plot_expanded_partition_batch.py](plot_expanded_partition_batch.py) 审计两个实机组的完成量、持卡模式和释放状态，从实机与回放结果分别生成彩图。
- [plot_ray_startup_review.py](plot_ray_startup_review.py)：从`results/ray-startup-review-20261125/evidence.json`读取已有失败时间戳，绘制进程创建、注册与actor就绪的彩色时间线。在仓库根目录执行`python -I docs/env-worker-deep-dive/env-service-verification/scripts/plot_ray_startup_review.py`，输出experiments下的`ray-startup-stages.png`与同名SVG，不启动Ray或GPU任务。

- [未知时长准入实验脚本](admission_study/README.md)
- [性能测量干扰实验脚本](feedback_pressure/README.md)
- [联合 GPU 放置实验脚本](joint_gpu_study/README.md)
- [PostTrain 裁判复用实验脚本](posttrain_reuse_study/README.md)
- [常驻模型计算共享实验脚本](resident_sharing_study/README.md)
- [状态复用实验脚本](state_reuse_study/README.md)
- [热状态放置实验脚本](warm_router_study/README.md)
- [agent_scheduler_brain_demo.py](agent_scheduler_brain_demo.py)
- [analyze_async_gpu_wait.py](analyze_async_gpu_wait.py)
- [analyze_capacity_validation.py](analyze_capacity_validation.py)
- [analyze_elastic_feedback.py](analyze_elastic_feedback.py)
- [analyze_elastic_partition.py](analyze_elastic_partition.py)
- [analyze_environment_capacity.py](analyze_environment_capacity.py)
- [analyze_flashinfer_racing.py](analyze_flashinfer_racing.py)
- [analyze_gpu_capacity.py](analyze_gpu_capacity.py)
- [analyze_gpu_control.py](analyze_gpu_control.py)
- [analyze_handover_overlap.py](analyze_handover_overlap.py) 从存档或本次运行读取采样与交接时间。
- [analyze_integrated_partition.py](analyze_integrated_partition.py)
- [analyze_low_load_path.py](analyze_low_load_path.py)
- [analyze_native_training.py](analyze_native_training.py)
- [analyze_natural_gpu.py](analyze_natural_gpu.py)
- [analyze_natural_gpu_fix.py](analyze_natural_gpu_fix.py)
- [analyze_paired_router.py](analyze_paired_router.py)
- [analyze_resident_matrix.py](analyze_resident_matrix.py)
- [analyze_roofline_sweep.py](analyze_roofline_sweep.py)
- [analyze_router_low_overhead.py](analyze_router_low_overhead.py)
- [analyze_science_tail.py](analyze_science_tail.py)
- [analyze_single_gpu_training.py](analyze_single_gpu_training.py)
- [architecture_walkthrough.py](architecture_walkthrough.py)
- [async_gpu_planning_demo.py](async_gpu_planning_demo.py)
- [async_return_gpu_probe.py](async_return_gpu_probe.py)
- [audit_baseline_state.py](audit_baseline_state.py)
- [audit_gate_contracts.py](audit_gate_contracts.py)
- [audit_remaining_env_design.py](audit_remaining_env_design.py)
- [audit_router_baseline.py](audit_router_baseline.py)
- [audit_science_command_readiness.py](audit_science_command_readiness.py)
- [audit_score_service_assets.py](audit_score_service_assets.py)
- [audit_swe_serve_candidates.py](audit_swe_serve_candidates.py)
- [auto_reservation_study.py](auto_reservation_study.py)
- [bpf_backend_workload.py](bpf_backend_workload.py)
- [bpf_cuda_identity.py](bpf_cuda_identity.py)
- [bpf_process_identity.c](bpf_process_identity.c)
- [calibrate_environment_capacity.py](calibrate_environment_capacity.py)
- [calibrate_roofline_bounds.py](calibrate_roofline_bounds.py)
- [candidate_source_version_probe.py](candidate_source_version_probe.py)
- [capacity_validation_pipeline.py](capacity_validation_pipeline.py)
- [check_cleanup.py](check_cleanup.py)
- [check_fixed_work_retry.py](check_fixed_work_retry.py)
- [check_imports.py](check_imports.py)
- [collect_training_evidence.py](collect_training_evidence.py)
- [command_context_probe.py](command_context_probe.py)
- [command_gap_cost_demo.py](command_gap_cost_demo.py)
- [command_group_objective_demo.py](command_group_objective_demo.py)
- [command_predictability_demo.py](command_predictability_demo.py)
- [compose_merge_check.py](compose_merge_check.py)
- [cross_pool_admission_demo.py](cross_pool_admission_demo.py)
- [cuda_driver_probe.py](cuda_driver_probe.py)
- [cuda_job_workload.py](cuda_job_workload.py)
- [cuda_pressure.py](cuda_pressure.py)
- [cuda_timing_probe.py](cuda_timing_probe.py)
- [dynamic_command_placement_demo.py](dynamic_command_placement_demo.py)
- [elastic_feedback_study.py](elastic_feedback_study.py)
- [elastic_partition_study.py](elastic_partition_study.py)
- [env_admission_attribution_demo.py](env_admission_attribution_demo.py)
- [env_reclaim_contract_demo.py](env_reclaim_contract_demo.py)
- [episode_offload_cost_demo.py](episode_offload_cost_demo.py)
- [explain_flashinfer_results.py](explain_flashinfer_results.py)
- [explain_gpu_policy_axes.py](explain_gpu_policy_axes.py)
- [explain_idle_window.py](explain_idle_window.py)
- [explain_inferencebench_score.py](explain_inferencebench_score.py)
- [explain_resident_admission_window.py](explain_resident_admission_window.py)
- [explain_resident_command_lifetime.py](explain_resident_command_lifetime.py)
- [explain_same_container_lock.py](explain_same_container_lock.py)
- [explain_unit_capacity.py](explain_unit_capacity.py)
- [fit_decode_capacity.py](fit_decode_capacity.py)
- [flashinfer_batch_experiment.py](flashinfer_batch_experiment.py)
- [flashinfer_batch_worker.py](flashinfer_batch_worker.py)
- [flashinfer_env_probe.py](flashinfer_env_probe.py)
- [flashinfer_official_eval.py](flashinfer_official_eval.py)
- [flashinfer_racing_probe.py](flashinfer_racing_probe.py)
- [flashinfer_split_experiment.py](flashinfer_split_experiment.py)
- [flashinfer_split_worker.py](flashinfer_split_worker.py)
- [gate_b_experiment.py](gate_b_experiment.py)
- [gpu_busy.py](gpu_busy.py)
- [gpu_command_handoff.py](gpu_command_handoff.py)
- [gpu_command_trace.py](gpu_command_trace.py)
- [gpu_conflict_experiment.py](gpu_conflict_experiment.py)
- [gpu_episode.py](gpu_episode.py)
- [gpu_job_probe.py](gpu_job_probe.py)
- [handover_overlap_followup.py](handover_overlap_followup.py)：固定任务与实时负载查询，仅切换交接重叠；每组 2 step、2700 秒上限，失败不自动重跑。
- [handover_timeout_followup.py](handover_timeout_followup.py) 复制首次重叠组快照并核验哈希，使用 285 秒单命令和 1200 秒轨迹期限补跑两步。
- [gpu_lending_demo.py](gpu_lending_demo.py)
- [harbor_gpu_check.py](harbor_gpu_check.py)
- [harbor_gpu_command_probe.py](harbor_gpu_command_probe.py)
- [harbor_shell_contract.py](harbor_shell_contract.py)
- [hold_gpu_capacity.py](hold_gpu_capacity.py)
- [host_proc_diagnostic.py](host_proc_diagnostic.py)
- [integrated_partition_agent.py](integrated_partition_agent.py)
- [integrated_partition_training.py](integrated_partition_training.py)
- [isolated_training.py](isolated_training.py)
- [joint_schedule_demo.py](joint_schedule_demo.py)
- [krex_measurement_demo.py](krex_measurement_demo.py)
- [local_command_placement_probe.py](local_command_placement_probe.py)
- [make_patched_tests.py](make_patched_tests.py)
- [mlgym_dapo_layers_demo.py](mlgym_dapo_layers_demo.py)
- [model_concurrent_gpu_jobs.py](model_concurrent_gpu_jobs.py)
- [model_engineering_backlog.py](model_engineering_backlog.py)
- [model_gpu_hold_scope.py](model_gpu_hold_scope.py)
- [model_gpu_suspend_cost.py](model_gpu_suspend_cost.py)
- [model_rollout_env_flow.py](model_rollout_env_flow.py)
- [model_router_recovery.py](model_router_recovery.py)
- [mps_capacity_demo.py](mps_capacity_demo.py)
- [mps_control_check.py](mps_control_check.py)
- [mps_cuda_client.py](mps_cuda_client.py)
- [mps_docker_probe.py](mps_docker_probe.py)
- [mps_parallel_check.py](mps_parallel_check.py)
- [mps_train_workload.py](mps_train_workload.py)
- [mps_training_analyze.py](mps_training_analyze.py)
- [mps_training_compare.py](mps_training_compare.py)
- [multinode_check.py](multinode_check.py)
- [node_prewarm_demo.py](node_prewarm_demo.py)
- [observe_science_containers.py](observe_science_containers.py)
- [one_step_science.sh](one_step_science.sh)
- [partition_followup.py](partition_followup.py)
- [partition_worker.py](partition_worker.py)
- [phase_backfill_demo.py](phase_backfill_demo.py)
- [placement_baseline_probe.py](placement_baseline_probe.py)
- [plot_async_return.py](plot_async_return.py)
- [plot_command_work.py](plot_command_work.py)
- [plot_gpu_capacity.py](plot_gpu_capacity.py)
- [plot_gpu_job_results.py](plot_gpu_job_results.py)
- [plot_router_control_ab.py](plot_router_control_ab.py)
- [plot_router_prewarm_ab.py](plot_router_prewarm_ab.py)
- [plot_science_budget_ab.py](plot_science_budget_ab.py)
- [plot_science_fixed_work.py](plot_science_fixed_work.py)
- [plot_science_pressure.py](plot_science_pressure.py)
- [plot_science_training_ab.py](plot_science_training_ab.py)
- [plot_startup_offload.py](plot_startup_offload.py)
- [plot_sync_choices.py](plot_sync_choices.py)
- [preparation_critical_path_demo.py](preparation_critical_path_demo.py)
- [prepare_flashinfer_batching.py](prepare_flashinfer_batching.py)
- [prepare_flashinfer_probe.py](prepare_flashinfer_probe.py)
- [prepare_flashinfer_racing.py](prepare_flashinfer_racing.py)
- [prepare_science_natural_gpu.py](prepare_science_natural_gpu.py)
- [prepare_verification_dependencies.py](prepare_verification_dependencies.py)
- [prepared_correctness_probe.py](prepared_correctness_probe.py)
- [prepared_score_probe.py](prepared_score_probe.py)
- [probe_subprocess_path.py](probe_subprocess_path.py)
- [process_identity_without_proc.py](process_identity_without_proc.py)
- [profile_nvml_sessions.py](profile_nvml_sessions.py)
- [profile_science_tail.py](profile_science_tail.py)
- [ray_actor_checks.py](ray_actor_checks.py)
- [ray_command_placement_probe.py](ray_command_placement_probe.py)
- [ray_score_preparation_probe.py](ray_score_preparation_probe.py)
- [real_admission_boundaries.py](real_admission_boundaries.py)
- [real_crash_recovery.py](real_crash_recovery.py)
- [real_env_worker.py](real_env_worker.py)
- [real_env_worker_gpu.py](real_env_worker_gpu.py)
- [real_env_worker_harbor.py](real_env_worker_harbor.py)
- [real_gateway_lifetime.py](real_gateway_lifetime.py)
- [real_gpu_containers.py](real_gpu_containers.py)
- [real_gpu_lending.py](real_gpu_lending.py)
- [real_gpu_preference.py](real_gpu_preference.py)
- [real_gpu_sharing.py](real_gpu_sharing.py)
- [real_harbor_check.py](real_harbor_check.py)
- [real_native_lifecycle.py](real_native_lifecycle.py)
- [real_resource_recovery.py](real_resource_recovery.py)
- [real_worker_supervision.py](real_worker_supervision.py)
- [resident_gpu_pressure.py](resident_gpu_pressure.py)
- [resume_capacity_validation.py](resume_capacity_validation.py)
- [review_native_harbor_lifecycle.py](review_native_harbor_lifecycle.py)
- [rollart_dynarl_reading_demo.py](rollart_dynarl_reading_demo.py)
- [rollout_command_verifier_demo.py](rollout_command_verifier_demo.py)
- [roofline_sweep.py](roofline_sweep.py)
- [router_live_recovery_check.py](router_live_recovery_check.py)
- [rsi_generation_handoff_demo.py](rsi_generation_handoff_demo.py)
- [run_command_context_probe.py](run_command_context_probe.py)
- [run_natural_gpu_capacity.py](run_natural_gpu_capacity.py)
- [run_science_command_work.sh](run_science_command_work.sh)
- [run_science_fixed_work.sh](run_science_fixed_work.sh)
- [run_science_pressure.sh](run_science_pressure.sh)
- [run_science_resident_matrix.sh](run_science_resident_matrix.sh)
- [scheduling_cost_demo.py](scheduling_cost_demo.py)
- [science_budget_ab.py](science_budget_ab.py)
- [science_budget_agent.py](science_budget_agent.py)
- [science_cuda_ab.py](science_cuda_ab.py)
- [science_fixed_work_agent.py](science_fixed_work_agent.py)
- [science_fixed_work_training.py](science_fixed_work_training.py)
- [science_native_backend_check.py](science_native_backend_check.py)
- [science_native_workload.py](science_native_workload.py)
- [science_natural_gpu_ab.py](science_natural_gpu_ab.py)
- [science_natural_gpu_agent.py](science_natural_gpu_agent.py)
- [science_natural_gpu_train.py](science_natural_gpu_train.py)
- [science_pressure_agent.py](science_pressure_agent.py)
- [science_pressure_audit.py](science_pressure_audit.py)
- [science_pressure_preflight.py](science_pressure_preflight.py)
- [science_pressure_training.py](science_pressure_training.py)
- [science_probe_agent.py](science_probe_agent.py)
- [science_router_control_ab.py](science_router_control_ab.py)
- [science_router_control_agent.py](science_router_control_agent.py)
- [science_training_ab.py](science_training_ab.py)
- [score_service_probe.py](score_service_probe.py)
- [shared_execution_check.py](shared_execution_check.py)
- [simulate_gate_b_composition.py](simulate_gate_b_composition.py)
- [summarize_command_placement.py](summarize_command_placement.py)
- [summarize_flashinfer_batching.py](summarize_flashinfer_batching.py)
- [summarize_preparation_windows.py](summarize_preparation_windows.py)
- [summarize_score_preparation.py](summarize_score_preparation.py)
- [swe_state_audit.py](swe_state_audit.py)
- [trainer_integration_check.py](trainer_integration_check.py)
- [unified_gpu_agent.py](unified_gpu_agent.py)
- [unit_budget_demo.py](unit_budget_demo.py)
- [unit_fragmentation_demo.py](unit_fragmentation_demo.py)
- [verifier_calibration_demo.py](verifier_calibration_demo.py)
- [verify_bpf_backend.py](verify_bpf_backend.py)
- [verify_bpf_gpu_identity.py](verify_bpf_gpu_identity.py)
- [verify_resident_admission.py](verify_resident_admission.py)
- [verify_unified_gpu.py](verify_unified_gpu.py)
- [versioned_placement_demo.py](versioned_placement_demo.py)
- [versioned_state_probe.py](versioned_state_probe.py)
- [vllm_env_sharing_demo.py](vllm_env_sharing_demo.py)
- [weight_copy_overlap.py](weight_copy_overlap.py)
- [yarn_feedback_contract_demo.py](yarn_feedback_contract_demo.py)

- [sync_overlap_study.py](sync_overlap_study.py)：冻结源码中的 NIXL 常驻同步对照，运行协议和边界见[同步实验](../../experiments/09-资源分区_静态卡池与推理余量共享.md#实验六权重同步能否提前完成)。
- [plot_static_partition_followup.py](plot_static_partition_followup.py)：从已有三组原始结果生成静态分卡比较图，在仓库根目录运行。

- [partition_prelaunch.py](partition_prelaunch.py)：使用冻结阶段画像筛选分区和容量，预测不读取本轮训练成绩。
- [partition_matrix_study.py](partition_matrix_study.py)：离线运行预先登记的四组配比验证，每组有界、失败停止、物理释放后交接。
- [evaluate_partition_matrix.py](evaluate_partition_matrix.py)：只读汇总矩阵，核对工作量、物理释放、冻结源码及大额次要 GPU 进程样本，可有界等待后台完成。
- [audit_sync_overlap.py](audit_sync_overlap.py)：只读审计同步与命令生存期、环境产物哈希及推算阶段，不启动 GPU 作业。
- [plot_partition_sync_study.py](plot_partition_sync_study.py)：从已有证据生成同步结果、时间线与配比协议 PNG／SVG，脚本可从任意目录运行。

配比研究在已激活的项目环境中运行，参数使用本次独立输出目录：

```bash
python docs/env-worker-deep-dive/env-service-verification/scripts/partition_matrix_study.py \
  --reference <已完成且物理清理通过的冻结对照目录> \
  --output <尚不存在的新实验目录>
```

执行前按集群运行指南核对私有节点、空闲 GPU、Docker、CUDA 与依赖来源。监督器只适用于本报告固定六卡任务，输出目录已存在时拒绝复用。`status.json` 记录进度，全部完成后输出 `analysis.json` 与 `prediction-validation.json`。具体四组、预算与已知预测误差见[实验报告](../../experiments/09-资源分区_静态卡池与推理余量共享.md#启动前预测选配比验证到了哪一步)。

- [plot_weight_storage_partition.py](plot_weight_storage_partition.py)：绘制运行时参数存储与静态分卡瓶颈解释图，只读取已有实验数据，不启动训练。

- [plot_partition_strategy_map.py](plot_partition_strategy_map.py)：从已完成证据绘制三类分区策略布局与新 3+1 配对的初始化／训练／清理时间图，在已激活项目环境中从仓库根目录运行；不启动 GPU 实验。

- [live_partition_study.py](live_partition_study.py)：冻结两步三组对照，待前序资源释放后离线执行；先用 `--repo --reference --output --wait-for` 准备，再用 `--run <输出目录>` 执行。仅在静态组训练已完成、内层物理释放通过、外层因已知探针导入错误停止且其余组从未启动时，可用 `--run <输出目录> --resume` 审计后续跑；它保存原始失败记录、复核冻结源码及当前 GPU，不重跑静态组，也不重试已启动的组。预算、分组与验收见[完整训练整卡验证](../../experiments/09-资源分区_静态卡池与推理余量共享.md#实验八训练途中把推理卡改给环境)。
- [plot_live_partition.py](plot_live_partition.py)：在项目环境运行，生成整卡调整、余量共享和版本恢复的 PNG／SVG，不启动训练。
- [plot_adaptive_admission_design.py](plot_adaptive_admission_design.py)：在仓库根目录运行，生成默认放满与暂缓第五条命令的教学时间线，输出到 experiments 的同名主题 PNG／SVG。数字为教学假设，不启动 GPU 任务，设计见[动态准入](../../knowledge/02-资源与调度/05-训推共享_vLLM余量与容量调度.md#默认放满并用完成时间反馈收紧准入)。

- [episode_partition_followup.py](episode_partition_followup.py) 从完成的 command 对照冻结源码准备 episode 静态池。参数为 `--reference <三组结果目录> --output <新组目录>`，保持工作量并审计基线策略差异，单组 2700 秒。`--trajectory-timeout-s 1200` 用于预先授权的延长单轨迹窗口，默认仍为 720 秒，旧结果不可覆盖。
- [audit_episode_partition.py](audit_episode_partition.py) 接收 episode 补测根目录，要求监督器已终止，核对完成工作量、持卡模式、快照和释放记录。失败或部分完成时不计算完整训练加速比。
- [audit_partition_startup.py](audit_partition_startup.py) 接收三组结果目录，读取历史日志、归档启动相关 Ray 摘录并生成 `startup-audit.json`。不启动实验，原始日志时区按本轮节点的 Asia/Shanghai 解析。
- [plot_handover_followup.py](plot_handover_followup.py) 绘制依赖导入、借卡窗口和首次交接对照。在有 Matplotlib 与中文字体的环境中，从仓库根目录运行 `python -I docs/env-worker-deep-dive/env-service-verification/scripts/plot_handover_followup.py --completed-only`，读取完整运行的工作量和交接记录，生成报告图 4 的 PNG 和 SVG。绘图只读取现存结果。
- [plot_live_partition_results.py](plot_live_partition_results.py) 读取两步训练记录，生成当前五组对照、Episode 补测，以及实验八的历史启动和训练分段图。PNG 与 SVG 输出到 experiments 目录。生成历史启动分段图前先运行只读启动审计。
- [plot_partition_baseline.py](plot_partition_baseline.py)：解释统一 episode 静态池的设备角色，以及同一 A、B 任务在 episode 与 command 下的预约时间线。激活项目环境后，在仓库根目录运行 `python -I docs/env-worker-deep-dive/env-service-verification/scripts/plot_partition_baseline.py`，生成 experiments 中的 `partition-episode-baseline.png` 和 SVG。图中时间是教学假设，不读取 GPU 或启动训练。
- [plot_env_reading_guide.py](plot_env_reading_guide.py)：生成 `env-reading-guide` 阅读导航和 `env-current-layout` 当前执行布局，PNG 与 SVG 保存在 experiments。激活项目环境后，在仓库根目录运行 `python -I docs/env-worker-deep-dive/env-service-verification/scripts/plot_env_reading_guide.py`。图解释文档职责和已配置路径，不报告现场利用率，不启动 GPU 服务或训练。

维护中的 `integrated_partition_training.py` 将预检、Ray 初始化、导入预检、配置预检和 driver 启动时间点写入 `status.json` 的 `startup_events`，供后续冻结实验区分启动等待。已完成及正在运行的历史快照保留原状。

后续新建的隔离 Ray 使用 `--ray-worker-register-timeout-s 90`，也可显式选 30 秒用于对照，整体初始化预算不变。它只影响本 runner 自己创建的 Ray，不修改共享集群。注册窗口加大是慢启动容错，不等同于性能优化。

- [observe_ray_startup.py](observe_ray_startup.py)：给定实验结果根目录及 `--py-spy <可执行文件>`，仅从本组 supervisor 的子进程采样尚在启动的 Python worker。默认最多 1200 秒，使用非阻塞栈采样，不记录局部变量，不暂停或终止模型进程。
- [check_ray_registration.py](check_ray_registration.py)：接收输出 JSON 路径与 `--node-ip`，在独立零 GPU Ray 实例验证 90 秒注册窗口和真实 CPU worker 注册，必须在前序实验物理释放后执行，并给外层进程设置时间预算。
- [plot_borrow_preparation.py](plot_borrow_preparation.py)：生成提前准备借卡的同工作量教学时间线，比较排队、命令完成与整批完成时刻，不启动真实任务。

- [startup_partition_followup.py](startup_partition_followup.py)：接收 `--reference <已完成的 elastic 组目录> --output <新目录> --repo <仓库>`，只覆盖已审计的启动入口和启动器，保留两步工作量、command 调度、720 秒轨迹期限及数据。独立组预算 2700 秒、初始化 1200 秒、一次尝试，自动核对快照、退出和物理释放。不实现排空期间的并行准入。
- [plot_partition_overlap.py](plot_partition_overlap.py)：生成 `partition-overlap-proposal.png/.svg`，解释模型退出时按已确认余量逐步接纳环境命令的设计提案。运行 `python -I docs/env-worker-deep-dive/env-service-verification/scripts/plot_partition_overlap.py`，只绘制教学数据，不启动 GPU 工作。
