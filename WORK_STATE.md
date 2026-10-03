# E063 current story pipeline — X1/X2 active

X0 complete and committed(a77a806). X1 jobs182228–182231 are the only new fits; X2 jobs182237/182238 are the only reduced-step evaluations. Ledger runs/story_round/jobs.json. New story_controller.py owns later X1 assembly/evaluation, X3 pilot/full, X4 covariance/dose, X5 seeds8/9. Do not restart legacy dispatchers or duplicate these jobs. X2 FP1-step exactFP32 fold and strict256-heldout gate passed; full500-episode result pending. See results/story_progress_20261003.json for the timestamped snapshot. Lower work states are historical.

# Current authority — 2026-10-03 evening story experiments

See STORY_2026-10-03.md. X0 COMPLETE: BF16-backbone activations with int8 embedding for both experts; M3 MSE anomaly remains unexplained. results/story_x0.json verifies the provenance. X1–X5 are newly authorized, with X1 then X2 binding GPU priority, X5 seeds8/9 allowed after X1–X2 queued. New ledger runs/story_round/jobs.json will own this round; never restart any historical controller. Lower sections are historical.

# Current result — FlowVQ main-method end-to-end validation complete

Both quantized-backbone FlowVQ policies now have strict reload, paired held-out MSE, independent safetensors accounting, and full four-suite seed7 validation. The authoritative five-row result is [MAIN_METHOD_TABLE.md](MAIN_METHOD_TABLE.md); all paired intervals and source hashes are in [complete evidence](results/flowvq_main_complete.json). The requested table is complete; no deferred experiment is authorized or dispatched by this controller.

# E061a current status — 2026-10-03: 500/2000 per policy; scheduler recovered

Both FlowVQ full runs remain incomplete: each has 500 validated unique episodes (worker 0), with no rollout error rows. Jobs 181202/181203 completed with exit 0:0 in 5h15m/5h07m. At the initial audit, the remaining three paired jobs had never started under preemptible priority. After recovery, pairs 181469 and 181470 are RUNNING on four L40S GPUs; pair 181471 waits for normal-QoS job slots. CPU collector 181472 is RUNNING and the ledger reports full_running without an error. No final medium/full result, paired bootstrap, or main-method table is available.

Controller 181214 failed at 2026-10-03 02:38:48 Singapore when Slurm's accounting database refused a connection. This interrupted monitoring; the two GPU workers completed normally. The controller now caches accepted successful jobs and retries scheduler query failures without submitting duplicate workers or weakening completeness checks. Both controller tests pass, including simulated outage recovery and one continuation per shared job.

Only verified PENDING jobs 181211/181212/181213 were canceled after the site rejected an in-place QoS change. Replacements 181469/181470/181471 use normal rose QoS and preserve worker assignments, checkpoints, and the frozen runtime/source contract. This QoS permits two paired jobs / four GPUs concurrently. New CPU collector 181472 resumes `runs/flowvq_main/jobs.json`; it creates and commits the final table only after both 2000-episode protocols complete. Do not submit duplicates. All deferred branches remain unlaunched. See the [progress snapshot](results/flowvq_main_progress_20261003.json).

# E060 current authority — 2026-10-03 main-method-only validation

The user requested only the five-row main-method table. [MAIN_METHOD_2026-10-03.md](MAIN_METHOD_2026-10-03.md) supersedes previous dispatch rules. Do not restart legacy controllers or submit any deferred branch.

Two new deployable checkpoints `bb_M3_flowvq_fp32` and `bb_M2_flowvq_fp32` are assembled and independently accounted. Strict reload / retained identity / held-out 256 observations from 40 disjoint trajectories all passed. Relative MSE is 0.007308567466560175 / 0.0429168180781574, against matching uniform 0.01304601522565916 / 0.037216964264135996. Both full evaluations are still required irrespective of MSE direction. Offline jobs 181174/181175 COMPLETED 0:0.

Live full jobs: 181202 (M3 worker0), 181203 (M2 worker0), paired jobs 181211 (M3 worker1 / M2 worker2), 181212 (M3 worker2 / M2 worker3), 181213 (M3 worker3 / M2 worker1). CPU collector 181214. Eight logical workers, at most eight GPUs; at dispatch only two GPUs were allocated, paired jobs pending. Strict runtime checks passed and real new episodes were recorded for both worker0 runs.

Only new ledger `runs/flowvq_main/jobs.json` owns this work. Full runner `scripts/run_flowvq_main.py` uses four disjoint workers per composition, seed7, four suites, 50 episodes/task. No existing matching medium episodes were found; medium is derived from matching first25 full IDs. Controller `scripts/flowvq_main_controller.py` validates all requested evidence and commits `MAIN_METHOD_TABLE.md` only after both complete. Read live ledger/Slurm before resubmission; total authorization <=8 GPUs. A final table is not yet available.

# E059 latest state — 2026-10-03T00:53:59+08:00 — CURRENT EVENING PIPELINE COMPLETE

All tasks in the current evening addendum finished successfully. FinalGPU180151 ended22:43:40SingaporeOct2, CPU180278 ended22:44:40; bothCOMPLETED0:0. squeue empty; ledger phase complete_authorized_pipeline, parallel_results p4/full/ablations true. Do NOT restart oldcontrollers or submit duplicates. No new experiment direction has been authorized beyond the completed conditional pipeline; wait for user direction on refinements/externalbaselines.

Finalmedium counts FP970,M1941,M2968,b*963,c962,d966 /1000 each. Strictcollector rerun,229sources verified,d1000uniqueIDs/966success independently counted. results/evening_ablations_medium.json final. d readratio1.121323>1.1budget, MSE.00448408(61.55%worse thanc), success+.4ppvs c => incrementalcriterionnotmet. b*only point-estimate jointtargetpass(+2.2pp,1.098345reads); allpairedrecoveryCIs cross0. UniformM2 remains highestpointestimateamongexpertvariants96.8%, no significantrankclaim. FullM3backbone/M2expert result97.4%/2000 confirmed; M2backbonefull notrun; P4onlyofflinecomplete. New completionrecord results/evening_completion_20261003.json. Reportsupdated withoutchangingruntime/model/protocol.

# E058 latest state — 2026-10-02T22:08:27+08:00

Full M3backbone+M2expert+FP32tables+int8 COMPLETE/PASS1948/2000 vsFP1928, +1pp CI[-.35,+2.65]; source49hashes checked. All4parallel GPUworkersCOMPLETED0:0, P4 allthree offlinecomplete (MSE.07653418/.11357073/.00741770), noP4rollouts. Controller180278 still RUNNING, parallel_results p4/full true; do not restart oldcontroller or submit duplicates.

b* andc medium COMPLETE:963/1000 and962/1000; matchingFP970,M1941,M2968. Strict separate summary results/evening_ablations_bc_medium.json plus results/evening_bc_comparison_frames.json;184hashes verified. b*targetpasses bypointestimate(+2.2pp,1.098345×reads), CIvsM1[-.9,+5.3]crosses0. c1.121323×reads fail1.1budget. d current881/1000 runningonoriginal2L40S180151; nofinaldmetric. Remainingestimated40–60min. Controllerwillwrite finalall-ablationaggregate afterjobfinish. Currentreportupdate changesno runtime source, gate, seed orschedule.

# E057 latest state — 2026-10-02T14:06:16+08:00

User requested parallel experiments. New active controller180278 uses scripts/evening_parallel_controller.py; oldCPU179698 was safely canceled while waiting. NEVER restart the legacy evening_controller.py: it would duplicate or serialize existing work. Same runs/evening/jobs.json and controller.lock; pre-handoff ledger preserved. Existing ablationGPU180151 remains running. New disjoint singleGPUworkers180284/180285/180286/180287 are assigned full worker0/1/2/3;180284/180285 now RUNNING P4offline,180286/180287 PENDING. First3 each do one P4offline warmup, then full; worker3 directlyfull. Maximum6GPUs<=8, actual4at14:07Singapore. P4 exports allcomplete, first2policy offline tasks running but no complete actionMSE yet. No P4rollouts.

Promotion now SELECTED bb_M3_ae_M2_fp32 after strict existingfullM2+mediumgates; lowerMSE rule unchanged despite M2backbone highermedium success. Full2000 uses1000validated medium+1000new; four disjoint workers, exactfrozencontract. Do not submit duplicatefull/P4 jobs. Worker failures retainlogs; completedrows reused. New source commits8d69beb and32f8878,7tests passed, existingruntime sources unchanged. Scheduler record results/evening_parallel_schedule_20261002.json. Independent summaries are produced as branchesfinish; no needwaitablations to startfull/P4. Currentcounts{"evening_b_star_M2": 500, "evening_c_M2": 147, "evening_d_M2": 0}.

# E056 latest state — 2026-10-02T13:46:46+08:00

Both deployment medium comparisons COMPLETE. `results/evening_deployments_medium.json`: M2backbone+M2expert982/1000 vsFP970 (+1.2pp, paired95%CI[-1.0,+3.8]); M3 counterpart974 (+0.4pp, CI[-1.5,+2.3025]). 94source hashes independently verified; strict collector rerun. Job179902 completed0:0 at11:53Singapore. Both pass, no mixed fallback. FP32tables/int8 unchanged. Lowest-heldout-MSE selection rule remains unchanged (M3 has lower MSE); no composition full job submitted or complete.

Ablations offline180133 complete; medium180151 running on2L40S, controller179698 waits. P4 policy offline pending, no rollouts. No new jobs or source changes. User explicitly requested report/results GitHub publication; E056 updates local REPORT/MODEL_MAP/EXPERIMENTS and complete aggregate, then synchronizes those four files to the public-reports repository. Earlier statuses below are historical.

# E055 latest state — 2026-10-02T11:07:18.309510+08:00

M3backbone+M2expert+FP32table+int8 medium COMPLETE/PASS:974/1000vsFP970, +0.4pp, paired95%CI[-1.50,+2.3025]pp. Dedicated summary `results/evening_deployment_M3_medium.json` validated all49source hashes; eventual controller two-variant summary unchanged. M2backbone medium 850/1000, no errors. LiveGPU179902 andCPUcontroller179698 run normally; no source or scheduling change in this turn.

c/d andP4 all layerfits complete, but full policyexports/offline/medium have NOTstarted; they are serialized after deployments_medium by the current controller. Do not describe fit completion as policy validation. M3-only full promotion has not occurred; no duplicate jobs submitted. PriorM2expert fullpass andgroup64standalone gate unchanged. Reports/aggregates updated in E055.

# E054 current state — 2026-10-02T09:35:52.348013+08:00

M2 full protocol COMPLETE/PASS:1923/2000=96.15%, FP1928/2000, -0.25pp, paired95%CI[-1.40,+0.75]. Expert default now M2. All98p1 result source hashes and raw counts independently verified. Keep existing frozen runtime sources.

Live GPU job179902 is deployments_medium on2L40S; CPUcontroller179698 waits normally. Snapshot M3+M2=886/1000, M2+M2=600/1000, no error files. Both complete screen contracts validated, crash-only. Offline MSE=.01304601522565916/.037216964264135996; b* .003971979837180978. No quantized-backbone full result or promotion yet.

Group64int4 standalone offline passes8.246323803397018e-5 and complete200screen193/200; current combinations keepint8 until combined validation. c/d all126layerfits complete, policyexport/offline/medium pending. P4 all3×126fits complete, actionMSE pending, no rollouts. The controller already owns all later stages; do not submit duplicates. No scripts/jobs modified in this status audit. Report-only synchronization includes new complete P1 and embedding aggregates.

# E053 latest scheduler state — 2026-10-01T16:32:03.084358+00:00

The live controller is **179698**, replacing 179687 solely to launch P4 fits as each c/d fitting pair releases its cards. The stopped CPU controller was verified to be waiting for the already queued initial offline job; no GPU worker, M2 episode or model artifact was interrupted. Source change `6eb1dbe` modifies scheduling only, outside the frozen rollout contract.

- P1 M2: 179604 running; summary 178967 unchanged.
- c/d: 179688 completed successfully (workers 0/1); 179689 running (workers 2/3). Last observation: c 121/126 layer artifacts, d 120/126, no policy-quality claim.
- New-composition/b*/group64 initial offline: 179695 pending for two L40S cards; reused by the new controller, no duplicate submission.
- P4 offline-only fitting: 179699 pending, 179700 depends on successful 179689. Each pair inherits its c/d pair's released slots; maximum four fitting GPUs plus two evaluation GPUs plus one P1 GPU remains seven.
- Public reports/aggregate exports were pushed and remote HEAD verified at `dae8e9209701af8556a490e1c0aee5b9d2941dcd`. The scheduler follow-up is being appended; live job ledger remains `/projects/yanlongvla/runs/evening/jobs.json`.

Do not restart 179687 or legacy backbone/P4 rollout dispatchers. Continue from the live ledger; no further user permission is required for these authorized stages.

# E053 audit supplement — 2026-10-01T16:31:14.853869+00:00

New independent reports: results/evening_checkpoint_accounting.json (M3/M2+expertM2+FP32tables whole3.638965/2.922590bpw, quality pending) and results/home_cleanup_20261002.json (combined7,560,856,033bytes safely removed). Cleanup manifests inhome preserved. Controller179687 alreadyrunning; initialoffline179695queued; do not duplicate. Initialexports complete. Activefit sources not modified during audit. See authoritative E053 below for pipeline details.

# E053 export follow-up — 2026-10-01T16:28:39.374080+00:00

Both new M2-expert backbone compositions andb* exported; header-derivedpayloads3.6389649302/2.9225899010wholebpw (M3/M2backbone), noqualityresultyet. Controller179687submittedinitialoffline179695 (2L40S). c/d fit179688 hasrealnewlayeroutputs;179689pending atlastcheck. Immutableimplementation `f82aeae`; livejobs at `runs/evening/jobs.json`. Publicreportonlypush `e1a0e78` verified; exportreportfollow-upbeing synchronized. A separately appearing untracked `results/home_cleanup_20261002.json` was not created by this turn and is intentionally preserved outside its commits. Our verifiedcleanup remains the pipmanifest listed below.

# E053 current override — evening addendum implemented and submitted

Snapshot 2026-10-01T16:26:04.805080+00:00; local implementation revision `f82aeae`. Latest authority is [ADDENDUM_2026-10-01_EVENING.md](ADDENDUM_2026-10-01_EVENING.md), with executable `configs/evening_oct1.json`. Older blocks below are historical where inconsistent. No new MSE/success claim yet.

- M2 P1 preserved: worker2 continuation179495 COMPLETED; final worker3 179604 RUNNING; summary178967 waits. Last inspected M2 count1821/2000, not a full result. Original P1 source contract matches all8manifests.
- Canceled only verifiedPENDING179609/179610/179481; allCANCELLED. Old backbone+expertM1 screening/full dispatcher disabled. Completed results preserved.
- New controller179687 RUNNING onCPU, command `sbatch --parsable --account=rose --qos=override-limits-but-killable slurm/evening_controller.sbatch`. Its exact child commands and state live in `/projects/yanlongvla/runs/evening/jobs.json`. Do not launch duplicate pipeline.
- c/d fits179688 workers0/1 RUNNING on2RTX5090,179689 workers2/3 PENDING. Controller concurrently assembles M3/M2backbones+expertM2+FP32tables+int8 andb*. It then launches2L40S offline workers (also group64int4gate), screens deploymentcrashes, andmedium1000comparisons. Fourlogicalrolloutworkers run2sequential/card.
- After c/d export and offline, mediumablation usesFP/M1/M2 matchingepisodeIDs. P4fitting usesreleased4fit slots thenoffline only. LegacyP4auto hook disabled. No P4rollout.
- Full promotion waits completeM2 ≤1.5ppfullgate andpassingnewcomposition. No backbone+M1promotion. Mixed2.5bpw fallback prepared; triggered by M2mediumfailure or explicitlyconfirmedmodelcrash, not infrastructurepreemption.
- Analytical read caveat: b*1.098345×M1, explicit-affinec/d1.121323×M1 >1.1target. Actualcminimumstorage81,913,068bytes vsM2 79,553,772 (+2,359,296). Theseareanalytical costs, noqualitybenefit/DRAMclaim. Reportexactoverheads inbothcomparisonframes.
- Group64int4 actual4.25embeddingbpw; originalBF16weights quantized; calibration1e-4gate mustpass before200screen. Deployment staysint8. FP32tabledefault; FP16failedmaxdiff0.0813788513 explicitlyinPAPER_NOTE.
- Seven targeted tests +syntax/whitespace passed. New sources mustremainfrozen once eveningrolloutmanifests exist; neitherP1 nor oldreference files overwritten. Allnew outputs aredistinctpaths.
- Homecleanupremoved753oldpipcachefiles,4,704,449,360bytes. Manifest `/home/yanlongc/home_cleanup_20261002.json`. Preserve allsource/data/env/checkpoint/log directories.

Next: inspectcontroller/workerlogs andactualSlurmstate; fixrealimplementationerrors whilepreservingartifactsandcontracts; publishonlyreviewedreports/aggregates. Do not fabricate completion whilejobsqueued.

# E052 current override — pending worker and backbone request recovery

## E052 — 2026-10-01T15:00:11.667826+00:00 — repeated preemption recovery; no new completed M2 result

At22:57Singapore expertM2=1762/2000 (spatial500,object500,goal428,libero10 334); noerrorrows. Workers0/1complete,worker2remaining46libero10,worker3remaining72goal+120libero10. Original178999 TIMEOUT14:45:56UTC; prearranged179495 resumed14:45:57UTC, savedepisodespreserved andrealnewrowsobserved. Worker3 job179000 Restarts3, pendingPriority since13:44:49UTC, sourceprotocol unchanged. Thus old2–3h optimisticestimate no longer reliable duequeue/preemption.

Verified179000PENDING, canceledwithscancel--state=PENDING andconfirmedCANCELLED before submitting sameworker3/variants/script to normalrose1L40S3h as179604. Summary178967 nowafterok179495:179604. Both areonlyactiveP1jobIDs; no overlap or rerun of completedepisodes. Ledger runs/method/p1_full_jobs.json updated.179604stillpending atcheck.

BackboneM2 still0/200,originalpairs179479/179480neverstarted, bothpending12hrequests. GuardedcanceledonlyPENDING andresubmitted identicalworkerpairs on samepreemptibleQoS with2h allocation(measuredM3~45–70min), replacements179609/179610. Summary179481 dependenciesupdatedafterok179609:179610. Ledger screen_jobs.json preservesoldcommands/history. No code/model/seed/ownership/contractchange; no duplicatepolicyjobs. Thisimprovesbackfillopportunity, doesnotpromiseallocation. ClusterbothL40nodesallocatedall8GPUs; onlyour179495currentlyrunningoneGPU. P4stillawaitsP1fullcompletion/resource release; P3alreadycomplete. No newscientificresult; public019cf85currentqualityresults unchanged.

# E051 current override — time-limit recovery scheduled

## E051 — 2026-10-01T13:34:57.637019+00:00 — ETA audit and P1 worker2 time-limit continuation

At21:32Singapore:expertM2 1527/2000. Suite-weighted remainingruntime ~86min worker2,~84min worker3 beforestartup/summary/queue. CompletedM2meanseconds/episode:spatial14.74,object17.85,goal13.80,libero10 31.61. BackboneM2screen0/200,179479/179480pendingPrioritywithnoSlurmstartestimate; M3screenmeansspatial26.70s/libero10 66.84s imply ~45–70min screeningoncefourGPUsallocated. These are estimates, not measured new results.

179000preempted/requeued once and resumed13:32:05UTC; savedepisodesretained. 178999remainingallocation~72min vs~86min estimatedwork; noTimeLimitmutationallowed. Submittedsameworker2/script/variants/protocol continuation179495 normalrose1L40S/3h afterany178999. Doesnotrunconcurrentlyorstartnewseeds; existingrunnerresumesmissingepisodeIDs. Summary178967 nowafterok179000:179495; completedworkers0/1alreadyvalidated. Strictcollectorstillrequiresall2000+source/reusecontracts, so afterany doesnotbypass quality/integritygates. Original178999runningunchanged; ledger runs/method/p1_full_jobs.json hasexactcommands/history. MaxP1twoactiveGPUs plusP2four<=6. P4waitsfullP1+P3asbefore. Latestpublicreport019cf85 remainscurrentqualityresults; no new quality outcome inthisETAturn.

# E050 current override — 2026-10-01T13:24:08.946184+00:00

## E050 — 2026-10-01 21:24（新加坡时间）: backbone M3 screening passes at exact threshold

179375 completed0:0 at13:18:43UTC; strictsummary179377 completed0:0 at13:19:12UTC. FrozenbackboneM3+expertM1:200/200pairedseed7episodes;190successes=95%,versusFP193=96.5%,difference-1.5pp. Spatial98/100vs97,libero10 92/100vs96. Success95%CI[90,99]%,paireddifference95%CI[-8,+5]pp,7improve/10regress. Required>=190 ismet exactly; no top1%Fisherrescue triggered. Source/read/storage/MSE unchanged (bb3.021028,allLinear2.791781,whole4.665800bpw,heldoutMSE.017078292602001867). Allcoverage/contractscheckedbyexistingpipeline; independentrawcount/suitesand30sourcehashesrecheckedhere. No extraGPUexperiment orseed.

AutomaticM2screenjobs179479(workers0/1),179480(workers2/3),summary179481 submittedaftergatepass; fullpromotion remainsafterbothdepthscreens byminimumheldoutMSEamongpasses. No backbonefullresultyet; passing200screen doesnotestablishequivalence and doesnoteraseexpertM1full-2.95pp. P1M2now1481/2000,summary178967awaitssingles178999/179000. P3complete;P4notlaunched. Report/aggregatepublication includesresults/backbone_M3_screen.json.

# E049 current override — 2026-10-01T13:13:05.859705+00:00

## E049 — 2026-10-01 21:13（新加坡时间）: M1 full-protocol result; P3 paired screen completed

M1 all2000episodes complete,40tasks/fourworker manifests/checkpoint/offline/sourcecontract/reuse200 unchanged validated by `python scripts/collect_promoted_full.py --root /projects/yanlongvla --variants a_M1 --output results/p1_M1_full.json`. OriginalFP1928/2000=96.4%; M11869/2000=93.45%; paired difference-2.95pp,95%CI[-6.10,+.30],59improvements/118regressions. Suites spatial481/500(-2pp),object468/500(-5.2pp),goal473/500(-2.2pp),libero10 447/500(-2.4pp). Independently recounted rawtotal/suites. Existing screening193/200 equality cannot support no-loss/equivalence wording; PAPER_NOTE/REPORT revised. No protocol, gate or bootstrap change. Complete M1-only file intentionally separate fromp1_full.json; M2notcomplete, no prematureP4dispatch.

P3job179385 COMPLETED0:0 at12:48:47UTC. Complete pairedint4screen190/200=95% (spatial98,libero10 92), pairedFPdifference-1.5pp,95%CI[-7,+3.5],5improve/8regress. Screenpassesbutoffline.0376305675 fails1e-4; int8fallbackunchanged. All28sourcehashesrevalidated. Results/reserved_validation.json complete; exactFP32folding/net469872600bytesaving unchanged. P3experimentphase complete independently ofP2.

LiveM2=1414/2000, M3backbonescreen=193/200 withnoerrorrows, neitherfinalyet. P2pair179376completed;179375waspreempted/requeued once(Restarts1), resumed12:49:02UTC using savedepisodes; no cancellation/resubmission by this turn. Summary179377awaits179375. P1singles178999/179000 nowM2;summary178967awaitsboth. P4 waitsfullP1resource release; prepareduniformpipeline, nofits. Actual4L40S, no additionalGPUjobsubmitted. Publicreports/aggregatesonly.

# E048 current override — 2026-10-01T11:27:31.258235+00:00

## E048 — 2026-10-01 19:27（新加坡时间）: P3 offline complete; exact FP32 folding, int4 rejected

Job178974 COMPLETED0:0,420.20seconds onL40S, all256calibrationobservations/noise0. FP16tables maxabs.08137885130593181 fails1e-6; FP32nativefull10x7actions are bit-identical(maxabs0). All37sites/10steps observation-independent. Removes474419200bytes; selectedtables+schedule4546600bytes, netsavings469872600bytes. Int8reserved542205320bytes/floor1.2934927974032224, versus old1012077920bytes. Int8relativeMSE1.995099412293451e-5 passes1e-4. Int4actualnibblepacking+FP16rowscale reserved278367368bytes/floor.6640771904268521, but relativeMSE.03763056753889389 FAILS; retainint8. Strictreloadandallfourraw/canonicalrepeatsPASS. Independent savednativearrays confirmexactFP32folding and canonicalarchives; both metrics recomputed and matched. Aggregate results/reserved_offline.json, implementation9961cb0. Calibration isolationusesoriginalteacher/nativeLinears; no combinedP2qualityclaim.

Followupint4pairedscreen179385queued(200episodesseed7), stillrequiredbyphase evenafterofflinefailure. M3backbonescreen179375runningtwoL40S,secondpair179376queued; summary179377afterboth. P1M1=1847/2000,M2=1100/2000 incl200reused. P1singles178999/179000running; summary178967awaitsfull. No fullsuccess/CI claimyet. P4unchanged,noGPUruns. Currentactual4L40S; publicscopeonlyreports/aggregates.

# E047 current override — 2026-10-01T11:16:22.627628+00:00 (2026-10-01 19:16（新加坡时间）)

P2 OFFLINE NOWCOMPLETE atM3/M2 afterdtypecomparisonrepair. Original178972failedfirstrepeat (rawFP64vsrecordedFP32), NOT qualitygatefailure. CompletedRHT256gatePASS2.011595740639109e-6 reused. Replacement179372 realnative-native+FP32-FP32repeat0 atall4guards, strictreload/retainedpass, full256heldout. M3MSE.017078292602001867,M2.043004072610800054, independentlyrecomputedarrays. results/backbone_offline/summary.json+table.csv,results/backbone_rht_validation.json. No LIBEROyet. Fitter/weights unchanged. Sourcefix b90383e,tests3pass; P3samefix+nativeFP64foldgate9961cb0 beforeP3start.

M3screenjobs179375(workers0/1),179376(workers2/3),summary179377submitted; verifylivequeue. 179131dispatcherCOMPLETEDafter179372. P3offline178974RUNNING onL40S, initialcaptureobs113atlastlog; noP3results. Itsint4screen autoafteroffline. P1job178947COMPLETEDbothworkers0/1M1/M2,178999worker2 and179000worker3RUNNING. M11813/2000,M21100/2000 incl200reuseeach, nofullstatsyet;178967summaryawaitsall. P4unchanged gatedautoafterP1+P3complete; adaptive at measuredkneestillrequiresimplementation.

P2M3gatefailurestillrequiresFP16top1%Fisherrescuebeforeothers, rescuecode notyetimplemented. No baselines/kernels/KD. Report/aggregateupdatedwithactualnewMSE; publishonlyreportfiles. No pendingCPUdiagnosticexec:optionalCPUtransformtest completedsuccessfully provingactualoutputFP64.

# E046 current override — 2026-10-01T08:58:12.957392+00:00 (2026-10-01 16:58（新加坡时间）)

P2 FITSAND EXPORTCOMPLETE. All288M3+288M2 artifacts,179014COMPLETED0:0 at07:50:38UTC,178966exportCOMPLETED0:0 at07:52:16UTC. models/hd_srvq_bb_M3_ae_M1 andM2 bothcomplete. Header-basedindependentpayloadaccounting results/backbone_checkpoint_accounting.json: bb3.021028/2.017084,allLinear2.791781/1.903451,whole4.665800/3.949425bpw. NoRHTguard/offlineMSE/screenyet; do notconflateexportwithqualitypass. NoactivefittingGPUs.

P1job178947RUNNING2L40S workers0/1nowM2nearfinish; thirdL40S178999RUNNINGworker2M1;179000worker3stillPENDING. CurrentM11122/2000,M21084/2000 incl200reuseeach. P1summary178967afterall. P2offline178972PENDINGQOSMaxJobsPerUserLimit(normalrosemax2jobs); shouldbeeligiblewhenP1pair178947releases. Dispatcher179131afteroffline, P3offline178974afteroffline. Do notrequeuefits orduplicateexport. P3/P4nonewqualityresults. No code/protocol changesinthisturn. Reportandmodelmapupdatedfornewstorageevidence; publishreportaggregatesonly.

# E045 current override — 2026-10-01T07:26:02.748220+00:00 (2026-10-01 15:26（新加坡时间）)

179056(worker2,A6000) and179057(worker3,RTX5090) VERIFIEDCOMPLETED0:0, each72layers atbothM3/M2. Activefit only179014(workers0/1,2RTX5090), Restarts=2, resumed07:24:12UTC. DoNOTcancel/resubmit: proposed guarded split didNOTexecute because jobbecameRUNNING; no mutationtofits/recipe. SavedM3/M2layers255/254 of288each. Logs reusedpreviouslayers andstartedmissingM2layer216/217.

Screen/full dispatcher NOW179131 afteroffline178972, screen_dispatch_needs_resubmit=false. Export178966stillafterremaining179014(and previouslycompleted179056/179057 satisfied), offline178972 thenP3offline178974. P1pair178947 RUNNING2L40S nowM2, firsttwo logicalworkers M1done; P1singles178999/179000 stillPENDING, summary178967awaitsall. M11100/2000,M2509/2000 including200reused each. ActualGPUcount4currently, extraL40Squeue. P2newMSE/successnone; P3none;P4uniformpipelinepreparedbutnogpujobs. P4resourcegate/hook fromE044unchanged. Publicreport/aggregateupdatedthisturn, noimplementationchanges.

# E044 current override — 2026-10-01T05:59:33.564275+00:00

P2 fit jobs NOW179014(workers0/1,2RTX5090 RUNNING),179056(worker2,1A6000 RUNNING; verifiedsavedlayersreusedandindex202newfitstarted),179057(worker3,1RTX5090 pending). Verify live state. 178961 was PREEMPTED at05:46 after50layers/worker/depth, requeued then canceledwhilePENDING; replacement179055 also canceledwhilePENDING tosplitremainingworkers. Allsavedlayerartifacts retained; same frozenrecipe 6e9f152f6bfbc879cf6349730b2d4dee5040e7cadd9cfffca3f14a41624ed813. NO duplicate oldjobs/resubmissions. Export178966 afterok179014:179056:179057. CPUdispatcher179010 canceledwhilePENDING forpreemptibleQOSsubmitcapacity; screen_dispatch_needs_resubmit=true, restored automatically byworker2/export viaensure_backbone_dispatch whenfitsfinish/free slots. Do not submit duplicate. P2sourcefitrecipe files unchanged.

IMPORTANT corrected run_backbone_screen.py previously invoked oldserve_vq; now actualserve_backbone. Addedrunner sourcehash toP2contract BEFORE anyP2screening. Startupmock testsP2/P4pass. P1runtimeunchanged. P2M3screenfailure still STOPSordinaryM2/full and requires top1%FisherFP16+firstblockM3 rescue implementation; notyetneeded/measured.

P4uniform pipeline NOW COMPLETE incode (assemble_lowbit,lowbit_policy,evaluate_lowbit_offline,run_lowbit_screen,lowbit_pipeline +slurm). NO GPUjobs/results yet. ensure_lowbit_dispatch calledbycollect_promoted_full andsummarize_reserved_screen onlyafterP1fullANDP3pairedcomplete, thenafterokcurrentstage ensurescurrentGPUrelease; fournormalroseRTX5090 fits thenexport/offline/screen. <=4P4+4P2GPUs, noP1/P3remaining. Ifdispatchfails,deferredJSONlogged, scientificresultsnotinvalidated. Callsareidempotent/locked. P4matched-read adaptivecentroid/subset andfullpromotionstillpendingmeasuredknee; promotionsonlyafterP1/P2fullcomplete. No externalbaselines/kernels/KD.

TestsP2/P4startup2, lowbitdispatch1, lowbitnumeric4 PASSED plusnewPython/shellsyntax. SnapshotP1M1=903/2000 (200reused), M2=200/2000. P2M3/M2layers=199/199 of288each. No newpolicyqualitymetric. Publicaggregate results/phase_oct1_progress.json readyforpublication withreportonly.

Current live override 2026-10-01T05:39:01.454955+00:00: BOTH fitpairs178961/179014 RUNNING on4RTX5090; P1pair178947RUNNING on2L40S, extraP1singles178999/179000stillPENDING. Six GPUsactuallyactive, up toeightwhenextraL40Savailable. PublicHEAD8fed31bb48a5547aaed61b0465b5c3ff99fed066 verified; latestfullcalibrationcomplete andfittingprogresspublished. Local3a3258e. Allrepositoriescleanbeforethisstateupdate; no activeexecsessions afterturncleanup.

# E043 current override — full backbone calibration COMPLETE; fits running

Fullcalibration178962 completed normally: 256observations/noise0, all288backboneLinear,180canonicalmoments, bothsame-device native-forward andindependentlyzeroedgradientrepeat checks EXACT. Aggregate results/backbone_calibration.json. FiveGemmafinalprefixbranches(q/o/gate/up/down layer17) havezeroactiongradientandare explicitlyHessian-weighted; no wholesaleFisherfallback. Noquantizedbackbonepolicyqualityresultyet.

Active fitting jobs: **178961 workers2/3 on2RTX5090 RUNNING;179014 workers0/1 on2additionalRTX5090 pending/verifylive**. PlannedA6000job178960 canceled onlywhilePENDING/nooutputs afterschedulerpredicted20:40UTC(~15hwait). User's prior flexible-card/speed-first authorization used to move onlyunstarted shards toalreadyvalidatedCUDA12.8RTX5090. Samefrozenrecipe andinputs; max4fit+4policyGPUs. Newrequest6h, estimatedscheduler09:25UTC atlastcheck butactualreleasecanchange. AllpolicyqualitycomparisonsremainL40S. DoNOTresubmitoldA6000job. Firstreplacementafterok178962 rejectedbecausecompletedcalibjobwasagedoutofcontroller; verifiedsacctCOMPLETEDandbothfullmanifests, thensubmittedwithoutobsoleteSlurmdependency. Fitterstillvalidatesfrozencalibration/hashgates.

P2export178966 nowafterok179014:178961; offline178972afterexport; restoredscreen/fullCPUdispatcher**179010** afteroffline (screen_dispatch_needs_resubmit=false). RHTpolicyguard onM3offlineworker uses256CALIBRATION observations withsameM1expert/nativevsfoldedbackbone, relativeMSE<=1e-4 andretainedbyteidentity; requiredbeforeP2screening. Rawrecipe/transformfileswerenoteditedduringin-flightfit. M3screenfailedgate triggersrecordedmandatorytop1%FisherFP16+firstblockprotectionbeforeordinaryM2/full; rescueimplementationstillneededonlyifgatefails. OnpasschainM2thenbestpassingMSEfull. Fivezerogradientbranchesareexpectedunusedfinalprefixoutputs, notunstableKVgradients.

P1 activejobs178947(workers0/1,2L40S running),178999(worker2normal1L40S pending),179000(worker3preemptible1L40Spending), lasttwo6h; summary178967afterallthree. NoothernewP1jobs. 178948/178989/178990canceledwhilependingwithoutoutputs; historyledgerpreserved. All4logicalownership/reuseproofunchanged. P3offline178974afterP2offline andautomaticint4pairedscreenfollowup unchanged; NO P3qualityresults yet. P4genericgroup16/K16fitter/decoderpreparedandtested,NO GPUFIT/evaluation; group16conditional/subset+wholeexport/evalstillrequired. Noexternalbaselines/kernels/KD.

Snapshot 2026-10-01T05:34:39.211836+00:00: fittedlayersM3/M2=84/82 of288each; thesearelayerartifacts, notpolicyresults.

E042 additional P2 numerical gate: M3offlineworker first runs validate_backbone_rht.py on256calibrationobservations, nativebackbone versusfoldedbackbone withsameM1expert. RequiresrelativeexecutedMSE<=1e-4 andretainedtensorhashidentity beforeM3offline/screening. P2fitsalreadyrunningcancontinue; no heldouttoadjustRHT. Sourcefitter/transformfilesunchanged, preserving in-flight recipe hashes.

# E042 current execution override — P1–P3 automation prepared

P1 active logical worker jobs: **178947 workers0/1 (normal rose,2L40S);178999 worker2 (normal,1L40S);179000 worker3 (preemptible,1L40S)**. Original178948 was verifiedPENDING with no worker2/3manifest, then canceled solely to use the one currentlyfreeL40 immediately. Allworkerownership/variants/50episodeprotocol/reuseproof unchanged. Fullsummary178967 dependencies nowafterokallthreeactivejobs; per-suite pairedCIsimplemented. Firstpair remainsrunning, don't restartcompletedepisodes.

P2: full256calibration178962 on2RTX5090; old178957 firstobservation cancellation-checkfailure retained, exact independent-zeroedrepeatgatefixed. Afterok178962: fit178960(2A6000workers0/1) and178961(2RTX5090workers2/3), M3/M2all288layers. Export178966 afterbothfits, offline178972 afterexport (2L40S). InitialCPU screendispatch178979 canceledwhilePENDING to free a QoSsubmit slot for higher-priorityP1worker3. **runs/backbone/jobs.json screen_dispatch_needs_resubmit=true until fitterworker2 or export restores it via ensure_backbone_dispatch.py**; newjobID recorded there. Thisisautomatic, do notduplicate. Dispatcher launches4logicalworkers on2pairs ofL40S via preemptibleQoS once fitsareallcomplete; maxphysicalL40Savailable8 andNOextraP4fitjob yet. M3complete200screen gate>=190: failure recordsrequiredtop1%FisherFP16+SigLIPfirstblockprotection andSTOPSordinaryP2grid; implementthatrescueonfailurebeforecontinuing. Onpass,M2screenthenminimumheldoutMSEpassingvariant→full,200screenreused,1800new; P2sourcecontractpinsnewloaderinadditiontounchangedcontrol. P1eval/server sourcesunchanged. NOTE actualfitterbeforethisturnhasnotstarted; newdecoder/export/offlinecodehasunit/parsechecks butwholemodelmeasurementpending.

P3 offline178974 afterP2offline178972,1L40S. Capturesnative37sites×10actualEulerstepsandchecksidenticalconditioningacrossall256observations; FP16tablesmustmaxabs<=1e-6elseFP32fallbackvalidated; dropsall39conditioningmatricesandtheirbiases, retainsoriginaltrainedparameterdenominator. Packedint4uses2nibbles/byte,FP16storedscale; nativeint8fallback. Bothself-containedexportsstrictreloadplusfourrepeatguards. ThirdcontrollerintegrationtestPASS: capturedPolicycallablerebound, ten-stepcursorresetacrosschunks, alterednumstepsrejected, allconditioningweightsremovedfromstateandpayloadexact. Afterofflinecomplete, automaticallysubmitsone2L40Sjobforfourlogicalint4screenworkers(twoinsequenceperGPU),200pairedseed7episodes,thenstatsinsidejob. Bothofflineand>=190successrequiredtochooseint4; failurekeepsint8. WholepolicyP3verificationpending, noclaimedfloordropyet.

P4 configurations recordedinconfigs/p4_lowbit_plan.json: analyticalactualexpertbpw group16K256M1=.5435929732; group8K16M1=.5179064202 withnibblepacking; group16K256M2=1.0701081247. Thesearecostprojections, NOFITS/EVALUATIONS. P4generic group16/K16fitter+packeddecoder prepared and3numericchecksPASS,fit_lowbit_layers.pyentryprepared butNOjobs. Fulldeployment/evaluation andgroup16conditional/subset extensions andkneematchedREADSablationsstillrequired. Respect8GPUtotalincludingpotentialP2fourL40+P1/P3; noneadditionalfitlaunchedyet. P4fullpromotionsonlyafterP1/P2evaldone. P5draftinPAPER_NOTE.mdandpublicREPORT, single-seedpendingP1. Baselines/kernelsremainprohibitedbeforeP1–P3complete.

Publicreportverifiedac4ab2aec33dbbe1efdfc31ecf84e2a62346decf afterE042update. Localimplementation34208b0. Onlyreport/aggregatefilesmaypublish. Noagentsauthorized. No scientificresultbeyond priorreference screensandnewcalibrationimplementationpilotyet.

# Current handoff — E041, 2026-10-01 new phase

User supplied PHASE_2026-10-01.md, resolving the historical E040 block. P1 explicit M1 then M2 full, P2 backbone critical path concurrently, P3 exact conditioning folding/int4, P4 below1bpw after P2 launch, P5 paper draft. Earlier pending direction question is RESOLVED. Do not resubmit old adaptive chain. P1 reuse verified for both policies; jobs 178947(workers0/1) and 178948(workers2/3), each starts M1 then M2. P2 pilot178955 PASSED on both RTX5090 workers (2 real observations each, exact native-forward and repeated-gradient checks); full256 calibration178957 failed on cumulative-score cancellation check; fixed to compare independently zeroed scores with exact threshold unchanged; replacement178962 submitted, then 2A6000 fit178960(workers0/1) +2RTX5090 fit178961(workers2/3), both dependencies updated to afterok178962. Source scripts/calibrate_backbone.py must not change while calibration is active. See runs/backbone/jobs.json and runs/method/p1_full_jobs.json for commands. No P2 outcome yet. Single seed and up to8GPU authorization persists. Historical snapshots below are superseded.

# E040 — blocked audit satisfied; awaiting experiment-direction reply

The same missing research-direction decision has now persisted for THREE consecutive goal turns: E038(no-success-knee finding/report), E039(protocol/reuse implementation completed), E040(current revalidation). E039 was real progress; this revalidation finds no remaining independent experimental step within the frozen method-order/grid rules. This is not a live-process wait: squeue is EMPTY, both repositories areclean, allpriorjobs terminal. No user direction reply has arrived in the conversation; do not treat automaticgoalcontinuation as an answer to thependingchoice.

Authoritative evidence: results/method_reference/summary.json statuscomplete,successesM3/M2/M1=193/191/193 of200, allpass, knee=null andablation_nominal_bpw=[]. Adaptiveoffline/adaptivescreen/promotedfull resultfiles doNOTexist. Step4wholemodel andStep5kernels remainorderedafterthemethodtable; KDalsogatedafterPTQtable. Do notclaimgoalcompletion. PublicHEAD867c13b8f7f3b54c056d093396eb476136e03438verifiedagain; completeexistingresults accessible.

Blockingdecision: amendtheuntriggeredknee-gridruletoMSE-focused(b)–(d)ablationsatactual~1.054/2.044bpw(recommended), orfirstconfirmuniformM1 onfull4suite x50seed7. Previousasyncchoice remainsopen; no duplicatequestiontool. Eight-GPU/singleseed authorizationpersists, noadditionalresourcepermissionneeded. Blockedthresholdsatisfied; rootwillsetgoalstatusBLOCKEDnow. Onnewuserdirection/resume, rereadlivegoal/state, retainallcompletedreferenceevidence, andimplementchosenbranchwithoutrequestingthealready-grantedresource/publicationpermissionsagain.

# E039 continuation update

Previousgoalturn=PROGRESS (allreference screening/publication). Currentturn=PROGRESS (rollout-contract gap fixed andCPUreal-data reuse integration passed). Publicreport updated andremoteverified **867c13b8f7f3b54c056d093396eb476136e03438**; includesE039reuse summary, priorcomplete reference resultsunchanged. Same scopeclarification remainspending forsecondconsecutivegoalturn; do not callblockednow. Slurmverifiedempty; no experiment launched. No pending execsessions.
New rollout_contract.py pins97source/assets/installedmetadatafiles andchecks bothupstreamtrackedtrees remainclean at everycompatibilitycheck; legacy referenceanchor results/reference_rollout_contract.json source-auditedagainstscreencommit, preservesoldmanifests. Newrun_vq_screen manifests carrycontract. prepare_promoted_full andcollect_promoted_full requirecompatiblecontract. Fourunitchecks+CPUshadowrootrealM1integrationpassed (200rows,repeatedinitstable,tamperedrow/protocolrejected,originalfilesunchanged). Noqualitymetricchanged. Full_reuse_summary.json isapprovedreportaggregate; entireprotocolanchor stayslocal. Thisdoesnotresolveemptyablationgrid. Continuefromuser'sdirectionwhenitarrives; no autonomous thresholdrelaxation.

# Latest override — E038, 2026-09-30T10:52UTC

**ALL reference(a) screening is COMPLETE: M3/M2/M1=193/191/193 of200. NOsuccessknee, original adaptivegrid EMPTY.** Strict summary177859 passed, results/method_reference/summary.json. D3b predictedM1failurebutactualM1passes; reportpredictionfalsified, no extrapolatedknee. M1paired7improvements/7regressions,net0pp, conditionalCI−5..+5.5pp. Wholemodel still13.9271bpw at1.0303expertbpw.

**NO activeSlurmjobs from thisproject.** 177854/177855/177859 complete. 177907 ran10seconds thenbothworkers2/3 exitedonno-gridguard, nofits. VerifiedPENDING177906/177915/177916/177917/177920/177925 canceledbeforestart; ledger recordsstates/reason. Do NOTsubmitmissingsecondscreenpair orresubmitadaptivechainwithout settlingthenewdirection. The historicalorchestrationinstructionsbelow areSUPERSEDEDbythisparagraph. Noadaptivepolicyfit/result exists.

AsyncuserquestionPENDING: since alllevels pass andno>=3ppknee, continueMSE-focused(b)–(d)ablationsatprospectiveactual1.054/2.044bpw(recommended), orfirstconfirmM1full4suite? This is a necessaryscopeclarificationoftheuser'sknee-gridrule, not a GPUpermissionrequest. Resourceup to8GPUs/singleseed authorizationpersists. GoalACTIVE, notpaused/complete/blocked. Do independentreporting; do notinventa knee orrunamendedgridwithoutresolution.

Implementationpreparedthrough68fc112: full-runwrapper run_promoted_full.py alsoexists(prepare/evaluate/summarize), explicitlyrequirescompleteadaptivePTQtableanditsselectedpassingconfigs. IfuserchoosesM1fullfirst, thatwrapper'scurrentselectiongatewon'tapply; useprepare_promoted_full.py +run_vq_screen.py explicit4suites/50episodes +collect_promoted_full.py undera recordedreferencevalidationplan. Noextra200episode rerun.

CurrentlocalreportandjournalE038includeallreferenceoutcomes, emptygrid, falsifiednoise-thresholdprediction. AllfourM1serverlogs auditedactualquantizedcheckpoint/nativeL40S; fourguardobservationsdifferfromoriginalFP, excludinganunchangedFPguardaccident. Public latest **aff732e** publishedcompleteM3/M2/M1screening,no-knee decision andfailednoise-thresholdprediction; remoteHEAD verified. Priorpublic1bf56d2containedM3/M2only.

---

# Current handoff — 2026-09-30, after E037

Goal ACTIVE. Method-first expert PTQ table, two promoted single-seed full runs, whole-model, limited kernels, public reports remain required. D3/D3b, all uniform(a)fits and offline actions are COMPLETE: do not rerun. No agents authorized.

## Constraints
Read ../AGENTS.md, PHASE_2026-09-30.md, ADDENDUM_2026-09-30.md, CONFIG.md. Latest user: speed first, up to8GPUs, SINGLEseed. Calibration/offline noise0; LIBEROseed7; full4suites x50eps/task xseed7. Exception32clean draws for obs186 already done, do not broaden. Primary heldout5x7 global-relative action MSE versus ORIGINAL FP,256observations from40disjointtrajectories. Actual storedexpertbpw matching, every metadata tensor counted; reads separate. No baselines/D4. KD only after PTQ table, frozen codes, velocity-level codebook/scale update <=1GPUday, separate row. Kernel work after method table only requested paths.

## Current completed reference evidence
D3b complete all1000noisy+200reusedclean, strict gate177820 passed. Amplitudes.03/.1/.3/.6/1 successes197/194/194/187/174 outof200. OriginalFP193; pass>=190. Firstfail.6 MSE.003672679735475301; precedingpass.3 MSE.0005751541783343518, no interpolated threshold. Early robustD3steps0–7 right-censored evenat300; frozen mean-one upper-envelope weights each2.1502997885874615e-5, step8 .4735269034014421, step9 9.526301072615471. Mean-based preserved. D1INCOHERENT; D2subsetsON. All details/results in REPORT and EXPERIMENTS E026–E034.

RHT wholepolicy passed relativeMSE8.6581684545e-7; unchanged retained tensors. Uniform126x3 layer fits and assembled checkpoints complete. Expert bpwM3/M2/M1=3.056850548946496/2.0435929731889204/1.0303353974313447. Whole14.115285062045798/14.021185744139224/13.92708642623265. Three complete heldout relativeMSE3.107445962500992e-5/.00016195514607488535/.01067681545496013. NativeL40S exact FP32repeat checks, allserialized tensors strict andretainedtensors exact. D3b predictions recorded before screens:3/2pass,1fail, heuristic only.

M3 screening COMPLETE193/20096.5% (spatial99,long94), M2 COMPLETE191/20095.5% (98,93), validated via summarize_method_reference.read_variant with all4manifests/hashes/coverage. Both pass. Skip adaptive nominal3. M1 screen still finishing at this snapshot; do not infer knee from partialcounts. results/reference_M3_screen.json and reference_M2_screen.json committed. CPU177859 will require all600reference episodes before writing results/method_reference/{summary.json,table.csv} and exactgrid decision.

## Active orchestration and resources
Ledger runs/method/adaptive_candidate_jobs.json records exact sbatch commands; referenceledger runs/method/reference_evaluation_jobs.json. Normal rose: max2runningjobs/max5submitted/4L40; use2jobs x2workers. Assigned preemptible override-limits-but-killable allows extra4fittingGPUs within8total, max5submitted. Do not bypass L40-1 reservation or cancelotherjobs. Policy comparisons stay validated L40S. A6000 and isolatedCUDA12.8RTX5090 are FITTING ONLY (device/real-layer checks already passed).

- Reference screens177854(workers0/1) COMPLETED,177855(workers2/3)finishing. CPU strictreference summary177859afterany both.
- Candidate fits177906:2A6000workers0/1;177907:2RTX5090workers2/3; bothafterok177859. Outputmodels/hd_srvq_adaptive_candidates. Four static layer partitions, M1/2/3 layer candidates onlyif a selected lower-average budget can use them. Nevera fullnominal3 adaptivepolicy. No candidates have started at this snapshot.
- CPU177915 plans andexports afterok177906:177907 ->runs/method/adaptive_plans/summary.json andevaluation_manifest.json, fullmodels/hd_srvq_<variant>.
- Offline177916normalL40workers0/1 and177917workers2/3 afterexport, distribute DISTINCT variants; independentCPU177920 recomputesarrays/metrics and(c)/b5%gate ->results/adaptive_offline.
- Screen177925normalworkers0/1 afterok177920. **Still need submit secondscreenpair workers2/3 once normalrose submitcapacity frees; ledger has onlyfirstpair.** Then CPUadaptive_screen_summary.sbatch afterokbothscreens; addtoledger. All screening rows are required even ifc5%stop gate fails; no further refiningfailedcomponent.

## Adaptive implementation and fixed protocol
Math/runtime sources committed and tested. vq_adaptive exactminimumreadprefix; fit_conditional_subset fixed-codegroup8WLS, nativeBF16-aware affine, exactsavedcentroidreuse for both. conditional_vq_linear stores onlydeployedsubsetbooks +FP16alpha/beta, nativeBF16nonserializeddensecaches. vq_heterogeneous projects readercovariances into common step-mean GPTQ compensation directions, jointbeam8, reader-only WLS, <=3swap passes; fit_heterogeneous_layer3rounds andselectsbestFULLnativecovariance includinginitial. Bookproposals mustimprovebothgroup andfull loss; swapsfull loss. Independent identicaldecoder reducesexactlytoordinaryGPTQ.
Real q_projM2 initialpilot177882 selects initial/noimprovement butpassesstorage/forward; retained. Correctedpilot177896 PASSES: sharedfullcalibloss67.35897827148438->47.895103454589844 selectedround0, payload558090, serialized10stepformulasexact. These areimplementationchecks, NO method-policy benefit claim. Filesmodels/hd_srvq_implementation_pilots/d_M2_layer0_compensated/result.json. Candidatefitter explicitly requires thispassedpilot pluscompletedreference/D3b gates; recipehash/resumehashes protectweights/moments/preview/entry/reference.

Storageprotocol E035: originalM1payload40,109,292 cannotfitfullc/dminimum41,030,892 =1.0540096398555872bpw. IfM1selected, reportoriginalbudgetinfeasible and compare mainvariants atthat commonbudget with newlyallocated uniform-per-step(a); originala_M1 remainskneeevidence. Higherbudget usesoriginaluniforma actualpayload (M2=79,553,772). Stored-budget greedyproposal then exactcostDP picksclosestreachable/mincaliblossatcost, reportslossincreasevsunderspendinggreedy. MismatchBLOCKSexport; nopadding/unusedplanes. Realshape cost-onlyDP confirms plaincanexactlyreach41,030,892 andprefixc canreach79,553,772; d actualsubsetmetadata verifiedonlyafterfitting.

Underfrozen minimumcode-plane-readbudget, per-layerb has exactlyonefullMstep andallothers1; calibration choosesstep9 ALL126layers atM2/M3, b==b*. PrefixRDcache runs/method/calibration_prefix_rd complete. b*/b**usebstored-depthplan; reuseexactb*artifact/config. c_centroid/c_affine/c_bothholdfullcselectedstoredplanes/codes/masksfixed; metadata differandreported, mainfullc matchedtoa/b/d. dM1maskdegenerate butheterogeneouscoderecalibrationcanstilldiffer; neverdedupentiredjustbecausemaskscoincide. Atadjusted1.054b maymixM1/M2layers.

plan_adaptive_variants.py verifies all126xvariantcandidatecoverage/hash, exactcommoncost, semanticartifactidentityforaliases andactualfixedcodes/centroidtensorsforinternalablations. prepare_adaptive_deployments.py reusesoriginala_M2when appropriate, exportsnewv2checkpointswithallretainedtensors. evaluate_vq_offline.py workswithv2 andneedsnativeL40; strictreloadallstate, retainedidentity,4FP32repeats. run_adaptive_evaluation.py uses4workers, offlinevariantownership thenallvariantscreeningtaskownership. summarize_adaptive_offline.py independentlyrecomputes256arrays and5%gate. summarize_adaptive_screen.py requirescompletepaired200perphysicalvariant andproducesPTQtable/G2/promotion, handlesaliaseswithoutduplicatecredit. No-gridcasepropagateswithoutinventingablation.

Prospective promotionrule(CONFIG,pre-adaptiveoutcome): up to2passing MAIN configs (a/b/b*/b**/c_both/d), firstoneperselectedbudget; fillsparefromremainingpassinguniqueconfigs. Atknee successfirstthenMSE; otherbudgetMSEfirstthensuccess. Require>=190, deterministicnameties, aliasescountonce. Internalcablationsreportedbutnoextrafullslots. If<2passing,nofailedgatepromotion. prepare_promoted_full.py reuses200exactscreenrows, then run_vq_screen.py --episodes50 --suites libero_spatial libero_object libero_goal libero_10. Explicitfullguardrejects2suite50. collect_promoted_full.py validatesall2000rows/fourworkermanifests/reusedrows, pairedsingle-seedbootstrap. These fullhelpers preparedbutNOTRUN; need Slurmfullwrapper/orchestrationaftercompletePTQtable.

## Publication and filesystem
Public https://github.com/ForeverBlue816/yanlongvla, latest verified **1bf56d261a117a047d72220d6873ae8ddc6998dd** containsD3b,alluniformoffline andCOMPLETE M3/M2screen+budgetpreflight. It doesnotyetinclude M1completegrid orE037implementationjournal. PublishreviewedREPORT/EXPERIMENTS/MODEL_MAP+aggregateJSONCSV ONLY; no code/config/weights/rawrollouts/actionarrays. Earlierbroaderpublicationautoreviewrejected; scopedreportpushworks. Localresearchmain hasNOremote.
Allmodels/data/runsunder/projects/yanlongvla ->/projects/_hdd/yanlongvla. Source scripts/env.sh. Tooldefaultbindsandboxfails; useexec require_escalated conciseauthorizedresearchjustification. /envs/report/bin/python=numpy/scipy; /envs/openpi/bin/python=torch2.7.1cu126; /envs/fit-cu128/bin/python=5090only. Barepythonabsent. unittest discover toavoidinstalledtestsnamespacecollision. No activefunctions execsessions at handoff except anyexplicitlylistedfuturework.

## Next actions
1. WaitcompleteM1/177859, publishfullreference3/2/1screen andactualknee; updateREPORTpendingphrases/WORK_STATE. No duplicateuniformruns.
2. Submit missingscreenworkers2/3 andCPUcompleteadaptiveaggregator as normal/preemptiblecapacityallows, recordledger. Monitorfourcandidateworkersforrealdata/runtimeerrors, thenexport/strict256heldout; publishmeasuredinitialadaptiveMSE andcstoprulewithoutwaitingfullrollouts.
3. FullinitialPTQscreen table, twoeligiblepromotions(fullsingle seed), thenseparatefrozencodeKD<=1GPUday onlyaftertable. Wholemodel andrestrictedkernels remain later.
4. Whole-model currentreservedfloor(nativeprotectedAdaRMS/timeMLP) makes3/2.5 infeasible evenwithminimumindexplanes: reserved2.4144276 plusminimumindices gives3.2208565752bpw beforebooks. Referresults/reserved_floor. Futurepossible exactFP32constantfoldtime-onlyAdaRMS onfixed10nativeactualtimesteps is NOTIMPLEMENTED/VALIDATED; do notclaimlowerfloor orquantizeprotectedweights. Keeporiginaldenominatorifvalidated. Int4embedding/eligiblelargeint8 scenarios remainunvalidatedcounterfactuals.

Snapshot UTC 2026-09-30T10:49:41.622830+00:00

```
             JOBID      STATE       TIME  NODES NODELIST(REASON)
            177925    PENDING       0:00      1 (Dependency)
            177917    PENDING       0:00      1 (Dependency)
            177916    PENDING       0:00      1 (Dependency)
            177920    PENDING       0:00      1 (Dependency)
            177915    PENDING       0:00      1 (Dependency)
            177907    PENDING       0:00      1 (Dependency)
            177906    PENDING       0:00      1 (Dependency)
            177859    PENDING       0:00      1 (Dependency)
            177855    RUNNING    1:18:58      1 gpu-l40-2
```
