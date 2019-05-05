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