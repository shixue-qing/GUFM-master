clear all; close all; clc
%part_dir = './chair/';
%part_name = 'chair_0988';       %��
a=dir('F:\���ݼ�\shrec16\����off_mat��100\2\*.mat');
for i=1:length(a)

A = load(fullfile(( 'F:\���ݼ�\shrec16\����off_mat��100\2\'),a(i).name));
%A = load('./chair/chair_0988'); %��
%[r c]=size(A.VERT);
%TRIV = A.TRIV;
%m = A.m;
%B=A.VERT(1:1916,:);   %��
%VERT = B;
%[n k]=size(VERT);
model_evecs = A.xyz_kuosan_hks1(:,4:153);
%save(['F:\���ݼ�\shrec16\����off_mat��100\chair\',part_name,'.mat'],'TRIV','VERT','m','n');
save(['F:\���ݼ�\shrec16\����off_mat��100\2\',a(i).name],'model_evecs') ;
end