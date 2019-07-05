python train.py l1normals --add_points --gpu_id 0 --loss_type PlaneNormSegLoss --load_normals --load_planes --network_type 'RectNetSegNormals' --workers 8


 python test.py ./experiments/RectNetSegNormals_PlaneNormSegLoss_dist2plane --add_points --gpu_id 0 --load_normals --load_planes --network_type 'RectNetSegNormals' --workers 8 --test_list ./data_splits/original_p100_d20_test_split_top50.txt --save_results

python train.py dist2plane_weights --add_points --gpu_id 0 --loss_type PlaneNormSegLoss --load_normals --load_planes --network_type 'RectNetSegNormals' --workers 8 --checkpoint ./experiments/RectNetSegNormals_PlaneNormSegLoss_dist2plane/checkpoint_006.pth --only_weights



# -------------

python train.py v1 --add_points --gpu_id 0 --loss_type 'PlaneParamsLoss' --load_normals --load_planes --network_type 'RectNetPlaneParams' --workers 8

 python test.py ./experiments/RectNet --add_points --gpu_id 0 --load_normals --load_planes --network_type 'RectNetPlaneParams' --workers 8 --test_list ./data_splits/original_p100_d20_test_split_top50.txt --save_results


#  -----------------------
python train.py saturday_v1 --add_points --gpu_id 0 --loss_type PlaneNormSegLoss --load_normals --load_planes --network_type 'RectNetSegNormals' --workers 8 --encoder_weights ./experiments/RectNet_Revis_all_baseline/model_best.pth

python test.py ./experiments/RectNetSegNormals_PlaneNormSegLoss_saturday_v1 --add_points --gpu_id 1 --load_normals --load_planes --network_type 'RectNetSegNormals' --workers 8 --test_list ./data_splits/original_p100_d20_test_split_top50.txt --save_results



python train.py saturday_debug --add_points --gpu_id 0 --loss_type PlaneNormSegLoss --load_normals --load_planes --network_type 'RectNetSegNormals' --workers 8 --encoder_weights ./experiments/RectNet_Revis_all_baseline/model_best.pth


python prediction.py ./experiments/RectNetSegNormals_PlaneNormSegLoss_only_depth_no_points/checkpoint_005.pth --add_points --empty_points --gpu_ids 1 --image_folder ../datasets/panores/ --network_type RectNetSegNormals


python train.py debug --add_points --gpu_id 0 --loss_type PlaneNormSegLoss --load_normals --load_planes --network_type 'RectNetPlanes' --workers 8 --encoder_weights ./experiments/RectNet_Revis_all_baseline_500/model_best.pth

# ----------------------

 python train.py debug --add_points --gpu_id 0 --loss_type PlaneNormClassSegLoss --load_normals --load_planes --network_type 'RectNetPlanesSphere' --workers 8 --checkpoint ./experiments/RectNetSphereNormals_PlaneNormClassSegLoss_with_sphere_dirs/model_best.pth --only_weights


#  ##
python train.py debug --add_points --gpu_id 1 --batch_size 2 --loss_type PlaneNormClassSegLoss --load_normals --load_planes --network_type 'RectNetSphereNormals' --workers 8 --checkpoint ./experiments/RectNetSphereNormals_PlaneNormClassSegLoss_with_sphere_dirs/model_best.pth --only_weights


    revis_dxdy_1: 0.14032043516635895,revis_dxdy_2: 0.20483054220676422, revis_l1_dist_1_0: 10.0,revis_l1_dist_2_1: 10.0, revis_normal_similarity_1: 0.12097064405679703, revis_normal_similarity_2: 0.20016883313655853



python train.py spherical_all_together --add_points --gpu_id 0 --batch_size 8 --loss_type PlaneNormClassSegLoss --load_normals --load_planes --network_type  RectNetSphereNormals --workers 8 --loss_scales "Plane_dist_plane_loss:10.0, revis_l1_dist_1_0:10.0, revis_l1_dist_2_1:10.0"


 CUDA_VISIBLE_DEVICES=1 python train.py refinement_with_planes --gpu_id 0 --batch_size 6 --loss_type PlaneNormSegLoss --load_normals --load_planes --add_points --network_type 'RectNetPlanes' --workers 8 --checkpoint ./experiments/RectNetPlanes_PlaneNormSegLoss_refinement_with_planes/checkpoint_005.pth


 CUDA_VISIBLE_DEVICES=0 python train.py debug --gpu_id 0 --batch_size 2 --loss_type PlaneNormClassSegLoss --load_normals --load_planes --add_points --network_type RectNetSmoothSphere --workers 8 --checkpoint ./experiments/RectNetSphereNormals_PlaneNormClassSegLoss_with_sphere_dirs/model_best.pth --only_weights

 python run_experiments.py ./experiments/lisa/ --run_test --test_list ./data_splits/list_best_False.txt --save_results