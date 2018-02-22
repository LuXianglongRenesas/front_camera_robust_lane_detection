pc_6d = zeros(length(pc(:,1,:,:))*length(pc(1,:,:,:))*length((pc(1,1,:,1))),6);
%%
%for o = 1:length(pc_6d)
    
count = 0;

for i = 1:length(pc(1,:,:,:))
    for j = 1:length(pc(:,1,:,:))
        for m = 1:length(pc(1,1,:,1))
            count = count + 1;
            %for o = 1:length(pc_6d)
    
            pc_6d(count,1) = j;
            pc_6d(count,2) = i;
            pc_6d(count,3) = m;
            pc_6d(count,4) = pc(j,i,m,1);
            pc_6d(count,5) = pc(j,i,m,2);
            pc_6d(count,6) = pc(j,i,m,3);
              %pc_6d(:) = pc(i,j,m);
           % end
            
        end
    end
end

%% generating ply files 
P = pc_6d(:,1:3);
C = pc_6d(:,4:6);

num = size(P, 1);
header = 'ply\n';
header = [header, 'format ascii 1.0\n'];
header = [header, 'comment written by Chenxi\n'];
header = [header, 'element vertex ', num2str(num), '\n'];
header = [header, 'property float32 x\n'];
header = [header, 'property float32 y\n'];
header = [header, 'property float32 z\n'];
header = [header, 'property uchar red\n'];
header = [header, 'property uchar green\n'];
header = [header, 'property uchar blue\n'];
header = [header, 'end_header\n'];

data = [P, double(C)];

fid = fopen('test.ply', 'w');
fprintf(fid, header);
dlmwrite('test.ply', data, '-append', 'delimiter', '\t', 'precision', 3);
fclose(fid);

%% read ply
ptCloud = pcread('test.ply');
pcshow(ptCloud);















% pc_6d(:,:,:) = pc(:,:,:);