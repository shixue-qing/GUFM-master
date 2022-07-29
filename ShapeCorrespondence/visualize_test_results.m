%% Visualize Unsupervised Network Results
clear all; close all; clc

addpath(genpath('./'))
addpath(genpath('./../Tools/'))

mesh_0 = load('./faust_synthetic/第三类/003'); %Choose the indices of the test pair
mesh_1 = load('./faust_synthetic/第三类/005'); %Choose the indices of the test pair  null_dog6217/dog

% mesh_0 = load('./faust_synthetic/holes_dog6187/holes_dog_shape_1'); %Choose the indices of the test pair
% mesh_1 = load('./faust_synthetic/null_dog6187/dog'); %Choose the indices of the test pair  null_dog6217/dog

% mesh_0 = load('./faust_synthetic/dog_mat/dog_009'); %Choose the indices of the test pair
% mesh_1 = load('./faust_synthetic/dog_mat/dog_010'); %Choose the indices of the test pair  null_dog6217/dog

% mesh_0 = load('./faust_synthetic/holes_dog6187/holes_dog_shape_10'); %Choose the indices of the test pair
% mesh_1 = load('./faust_synthetic/null_dog6187/dog'); %Choose the indices of the test pair  null_dog6217/dog

% X = load('./Results/test_第三类_ms120/006.mat_003.mat.mat'); %Choose the indices of the test pair
% X = load('./Results/test_第三类_jjtz120_svd/008.mat_003.mat.mat'); %Choose the indices of the test pair
% X = load('./Results/test_第三类_jjtz120_coseig/008.mat_003.mat.mat'); %Choose the indices of the test pair
% X = load('./Results/test_第三类_ronghe150_cuo/006.mat_003.mat.mat'); %Choose the indices of the test pair
% X = load('./Results/test_第三类_ronghe150_cuo33000/006.mat_003.mat.mat'); %Choose the indices of the test pair
% X = load('./Results/test_第三类_ms120ronghe_tzxl30/008.mat_003.mat.mat'); %Choose the indices of the test pair

X = load('./Results/test_第三类_linyu_mscos_E120/003.mat_005.mat.mat'); %Choose the indices of the test pair
% X = load('./Results/train_第三类_linyu_mscos_E120_COS/003.mat_005.mat.mat'); %Choose the indices of the test pair

% X = load('./Results/test_第三类_ms120cg12/003.mat_005.mat.mat'); %Choose the indices of the test pair

% X = load('./Results/test_第三类_ms120E4/003.mat_005.mat.mat'); %Choose the indices of the test pair
% X = load('./Results/test_new3/003.mat_000.mat.mat'); %Choose the indices of the test pair

% X = load('./Results/test_第十四类_ms120/holes_dog_shape_000.mat_holes_dog_shape_010.mat.mat'); %Choose the indices of the test pair
% X = load('./Results/test_dogms120/dog_009.mat_dog_010.mat.mat'); %Choose the indices of the test pair

% X = load('./Results/test_faust_synthetic_holes_dog6187/holes_dog_shape_009.mat_holes_dog_shape_010.mat.mat'); %Choose the indices of the test pair

[~, unsupervised_matches] = max(squeeze(X.softCorr),[],1);

colors = create_colormap(mesh_1,mesh_1);
figure('color',[1 1 1]);
subplot_tight(1,2,1); 
colormap(colors);
plot_scalar_map(mesh_1,[1: size(mesh_1.VERT,1)]');
freeze_colors;
%title('Target');
axis off;

subplot_tight(1,2,2); colormap(colors(unsupervised_matches,:));
plot_scalar_map(mesh_0,[1: size(mesh_0.VERT,1)]');freeze_colors;
%title('Source'); 
axis off;
