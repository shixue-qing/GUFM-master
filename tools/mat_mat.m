clear all; close all; clc
% a=dir('G:\��ʮ����\*.mat');
% for i=1:length(a)
% load(fullfile(( 'G:\��ʮ����\'),a(i).name));
% TRIV = surface.TRIV;
% [m,c] = size(TRIV); 
% VERT(:,1) = surface.X;
% VERT(:,2) = surface.Y;
% VERT(:,3) = surface.Z;
% % shape.X = data(1,:)';
% [n,r] = size(VERT); 
% disp(a(i).name);
% save(['F:\���ݼ�\ģ��1\2011����\�½��ļ���\��ʮ����\',a(i).name],'TRIV','VERT','m','n','surface');
% end

a=dir('E:\������ʵ�ִ���\uns_dfm\uns_master\uns_master\Learning Correspondence of Synthetic Shapes\faust_synthetic\shapes\*.mat');
for i=1:length(a)
load(fullfile(( 'E:\������ʵ�ִ���\uns_dfm\uns_master\uns_master\Learning Correspondence of Synthetic Shapes\faust_synthetic\shapes\'),a(i).name));
surface.TRIV = TRIV;
surface.X = VERT(:,1);
surface.Y = VERT(:,2);
surface.Z = VERT(:,3);

disp(a(i).name);
save(['E:\������ʵ�ִ���\uns_dfm\uns_master\uns_master\Learning Correspondence of Synthetic Shapes\faust_synthetic\shapes_ms\',a(i).name],'TRIV','VERT','m','n','surface');
end