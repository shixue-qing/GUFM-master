% This file implements the method described in:
%
% "Consistent Partial Matching of Shape Collections via Sparse Modeling" 
% L. Cosmo, E. Rodola, A. Albarelli, F. Memoli, D. Cremers
% Computer Graphics Forum, 2016
%
% If you use this code, please cite the paper above.
% 
% Luca Cosmo, Emanuele Rodola (c) 2015

function shape = loadoff(filename)
%% Loads off model
%addpath(genpath('./shrec2016_PartialDeformableShapes/'))
shape = [];

%f = fopen(filename, 'rt');  %wolf.off
%filename = './shrec2016_PartialDeformableShapes/null/wolf.off';
f = fopen(filename, 'rt');
fgetl(f);
h = sscanf(fgetl(f), '%d %d %d');
nv = h(1); %4344
nt = h(2); %8684

data = fscanf(f, '%f');  %点的坐标 3 面片

shape.TRIV = reshape(data(3*nv+1:3*nv+4*nt), [4 nt])';  %面片行：最后
shape.TRIV = shape.TRIV(:,2:end)+1 ; %应该不+1

data = data(1:3*nv);  %点
data = reshape(data, [3 nv]);
% 
 shape.X = data(1,:)';
 shape.Y = data(2,:)';
 shape.Z = data(3,:)';
shape.VERT = data';
shape.m = nt;
shape.n = nv;
fclose(f);
