clear all; close all; clc
% a=dir('G:\第十四类\*.mat');
% for i=1:length(a)
% load(fullfile(( 'G:\第十四类\'),a(i).name));
% TRIV = surface.TRIV;
% [m,c] = size(TRIV); 
% VERT(:,1) = surface.X;
% VERT(:,2) = surface.Y;
% VERT(:,3) = surface.Z;
% % shape.X = data(1,:)';
% [n,r] = size(VERT); 
% disp(a(i).name);
% save(['F:\数据集\模型1\2011分类\新建文件夹\第十四类\',a(i).name],'TRIV','VERT','m','n','surface');
% end

a=dir('E:\各论文实现代码\uns_dfm\uns_master\uns_master\Learning Correspondence of Synthetic Shapes\faust_synthetic\shapes\*.mat');
for i=1:length(a)
load(fullfile(( 'E:\各论文实现代码\uns_dfm\uns_master\uns_master\Learning Correspondence of Synthetic Shapes\faust_synthetic\shapes\'),a(i).name));
surface.TRIV = TRIV;
surface.X = VERT(:,1);
surface.Y = VERT(:,2);
surface.Z = VERT(:,3);

disp(a(i).name);
save(['E:\各论文实现代码\uns_dfm\uns_master\uns_master\Learning Correspondence of Synthetic Shapes\faust_synthetic\shapes_ms\',a(i).name],'TRIV','VERT','m','n','surface');
end