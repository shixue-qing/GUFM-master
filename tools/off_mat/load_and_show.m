%% Copyright (c) Cosmo Luca - luca.cosmo@unive.it
%  SHREC2016 - Partial Matching Of Deformable Shapes
%  http://www.dais.unive.it/~shrec2016/
clear all; close all; clc

%addpath(genpath('./shrec2016_PartialDeformableShapes_TestSet/'))

%addpath(genpath('./../Tools/'))
%type = 'holes'; %dataset name [cuts|holes]
%base_dir = '../shrec2016_PartialDeformableShapes'; %directory conatining the datasets subfolders (cuts/holes/null) 

%null_dir = sprintf('%s/%s/',base_dir,'null');
%part_dir = sprintf('%s/%s/',base_dir,type);
%null_dir = './shrec2016_PartialDeformableShapes/null/';
%part_dir = './shrec2016_PartialDeformableShapes_TestSet/holes/';
part_dir = './sof/';

files = dir([part_dir,'*.off']);

for fi = 1:numel(files)  %1:79

    fprintf('-------------------------------------\n');
    fprintf('Loading shape %d of %d\n',fi, numel(files));
    
    fname = [files(fi).name]; %——1.off
    [~, part_name, ~] = fileparts(fname);
    
    %model = regexp(part_name,'[^_0-9]*','match');
    %model = [model{2}];
    
    %M = load_off([null_dir model '.off']);     %完整
    %TRIV = M.TRIV
    %VERT = M.VERT
    %m = M.m
    %n = M.n
    %save(['F:\数据集\shrec16\尝试off_mat+1\shrec2016_PartialDeformableShapes\null\',model,'.mat'],'TRIV','VERT','m','n')
    N = load_off([part_dir part_name '.off']); %残缺
    TRIV = N.TRIV;
    VERT = N.VERT;
    m = N.m;
    n = N.n;
     surface.X = N.X;
     surface.Y = N.Y;
     surface.Z = N.Z;
     surface.TRIV = N.TRIV;
    
    %save(['F:\数据集\shrec16\尝试off_mat+1\shrec2016_PartialDeformableShapes_TestSet\holes\',part_name,'.mat'],'TRIV','VERT','m','n')
    save(['F:\数据集\shrec16\尝试off_mat乘100\sofa\',part_name,'.mat'],'TRIV','VERT','m','n','surface');

   
end

