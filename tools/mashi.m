clear all; close all; clc

%% ����·��


a=dir('F:\���ݼ�\ģ��1\2011����\������zuo_linyums\*.mat');
for fi=1:length(a)
% fi = 1;    % ����Ҫ��1��ʼ�����ݼ��ܸ���    ��
load(fullfile(( 'F:\���ݼ�\ģ��1\2011����\������zuo_linyums\'),a(fi).name));             %��

% D = pdist2(VERT,VERT,'mahalanobis');cosine
O_juli = pdist2(D,D,'mahalanobis');

hmin=min(min(O_juli));
hmax=max(max(O_juli));
DD =(O_juli-hmin)*(1-0)/(hmax-hmin);
% D = pdist2(D,D);

save(['F:\���ݼ�\ģ��1\2011����\������zuo_linyums_ms\',a(fi).name],'DD');
disp(a(fi).name)
end


% clear all; close all; clc
% 
% %% ����·��
% 
% 
% a=dir('E:\������ʵ�ִ���\uns_dfm\uns_master\uns_master\Single Pair Experiment\artist_models\*.mat');
% fi = 2;
% % fi = 1;    % ����Ҫ��1��ʼ�����ݼ��ܸ���    ��
% load(fullfile(( 'E:\������ʵ�ִ���\uns_dfm\uns_master\uns_master\Single Pair Experiment\artist_models\'),a(fi).name));             %��
% 
% D = pdist2(model.VERT,model.VERT,'mahalanobis');
% % D = pdist2(D,D);
% 
% save(['F:\���ݼ�\ģ��1\2011����\ysj_ms1\',a(fi).name],'D');
% disp(a(fi).name)
