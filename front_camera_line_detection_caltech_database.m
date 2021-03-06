%% show the image.
%im = imread('front_straight.png');
tic;
tic;
I = imread('front_curved.png');
%I = imread('front_straight.png');
I = imresize(I, [900 600]);
t1 = toc;
disp(['STEP 1 read raw image t1 :' num2str(t1)]);

%% Image Crop and section initializations
% figure(1)
% subplot(1,3,1)
% imshow(I);
% hold on;
%draw ROI sections 
upper_bound = 355; lower_bound = 650;
line_x = 0:600;
line_y1 = upper_bound; line_y2 = lower_bound;
% plot(line_x,line_y1,'-o','MarkerSize',2,'Color',[1 0 0]);
% plot(line_x,line_y2,'-o','MarkerSize',2,'Color',[1 0 0]);

tic;
I_ROI = imcrop(I,[0 upper_bound 600 lower_bound-upper_bound]);
% subplot(1,3,2); 
I_ROI_resize = imresize(I_ROI, [600 600]);
t2 = toc;
disp(['STEP 2 ROI img crop t2:', num2str(t2)]);
% imshow(I_ROI_resize)

%cut ROI into 7 sections 
sec_wid_1 = 150;sec_wid_2 = 150;sec_wid_3 = 100;
sec_wid_4 = 100;sec_wid_5 = 50;sec_wid_6 = 25;sec_wid_7 = 25;
%draw those sections 
% hold on;
% plot(line_x,600-sec_wid_1,'-o','MarkerSize',2,'Color',[1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2,'-o','MarkerSize',2,'Color',[1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2-sec_wid_3,'-o','MarkerSize',2,'Color',...
%     [1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2-sec_wid_3-sec_wid_4,'-o','MarkerSize',...
%     2,'Color',[1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2-sec_wid_3-sec_wid_4-sec_wid_5,'-o',...
%     'MarkerSize',2,'Color',[1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2-sec_wid_3-sec_wid_4-sec_wid_5-sec_wid_6,'-o',...
%     'MarkerSize',2,'Color',[1 0 0]);
tic;
GS = rgb2gray(I_ROI_resize);
%simple color filtering here before sobel
%Bin_img = imbinarize(GS,'0.7');
Bin_img = GS > 160;
BW = edge(Bin_img,'Sobel');
t3 = toc;
disp(['STEP 3 grayscale and all filters:', num2str(t3)]);

% subplot(1,3,3)
% imshow(BW);
imwrite(BW,'temp_BW.png');
tic;
points = lsd('temp_BW.png');
t4 = toc;
disp(['STEP 4 LSD:', num2str(t4)]);
% hold on;
%apply LSD here
% for i = 1:size(points, 2)
%     plot(points(1:2, i), points(3:4, i), 'LineWidth', points(5, i) / 2, 'Color', [0, 1, 0]);
% end
% 
% plot(line_x,600-sec_wid_1,'-o','MarkerSize',2,'Color',[1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2,'-o','MarkerSize',2,'Color',[1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2-sec_wid_3,'-o','MarkerSize',2,'Color',...
%     [1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2-sec_wid_3-sec_wid_4,'-o','MarkerSize',...
%     2,'Color',[1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2-sec_wid_3-sec_wid_4-sec_wid_5,'-o',...
%     'MarkerSize',2,'Color',[1 0 0]);
% plot(line_x,600-sec_wid_1-sec_wid_2-sec_wid_3-sec_wid_4-sec_wid_5-sec_wid_6,'-o',...
%     'MarkerSize',2,'Color',[1 0 0]);

%% Classification of multiple lanes based on thres_dis and thres_angle
% study of matrix points (42 tiny line segments)
% figure(2)
% hold on;
% axes('pos',[0 0 1 1]);
% axis([0 10 0 10])
% %lines format [x1 x2 y1 y2 line_width] in matrix points 
% for i = 1:length(points)
% H = annotation('arrow',[points(1,i)/1000 points(2,i)/1000],...
%     [(600-points(3,i))/1000 (600-points(4,i))/1000]);
% %H = annotation('arrow',[.3 .6],[.3 .8]);
% end

%% Sorting line segment vectors (same direction)
tic;
points_2 = points;
for i = 1:length(points_2)
   if points_2(3,i) <= points_2(4,i)
       temp_y = points_2(3,i);
       points_2(3,i) = points_2(4,i);
       points_2(4,i) = temp_y;
       temp_x = points_2(1,i);
       points_2(1,i) = points_2(2,i);
       points_2(2,i) = temp_x;
   end
   
end
t5 = toc;
disp(['STEP 5 sorting line segment vectors:', num2str(t5)]);
% figure(3)
% hold on;
% axes('pos',[0 0 1 1]);
% axis([0 10 0 10])
% %lines format [x1 x2 y1 y2 line_width] in matrix points 
% for i = 1:length(points_2)
% H = annotation('arrow',[points_2(1,i)/1000 points_2(2,i)/1000],...
%     [(600-points_2(3,i))/1000 (600-points_2(4,i))/1000]);
% %H = annotation('arrow',[.3 .6],[.3 .8]);
% end

%% Radon transform (roughly detect how many lines are there)
tic;
theta = 0:180;
% figure(4)
[R,xp] = radon(BW,theta);
t6 = toc;
disp(['STEP 6 Radon Transform:', num2str(t6)]);
% imagesc(theta,xp,R);
% title('R_{\theta} (X\prime)');
% xlabel('\theta (degrees)');
% ylabel('X\prime');
% set(gca,'XTick',0:20:180);
% colormap(hot);
% colorbar

%% Local maximum approach
I_width = 600; %600x600 square binary img
ct_width = I_width/2;
hLocalMax = vision.LocalMaximaFinder;
      hLocalMax.MaximumNumLocalMaxima = 50;
      hLocalMax.NeighborhoodSize = [round(I_width/10+1) round(I_width/10+1)]; % must be odd number
      hLocalMax.Threshold = round(I_width/8); %6 or 12 based on images 
      local_max_pt = single(step(hLocalMax, R));
      
      
%% to find ONE start point of for each lane from LSD

%%
%%combining tiny line segments
%calculate angle of each line segment and add those to row6 of point_2
for i = 1:length(points_2)
    %points_2(6,i) = 180/3.1415*atan((points_2(4,i)-points_2(3,i))/(points_2(2,i)-points_2(1,i)));
    points_2(6,i) = (points_2(4,i)-points_2(3,i))/(points_2(2,i)-points_2(1,i));
end

dis_thres = 7; %threshold is 3 pixels 
dis_thres_count = 0;
clear index_pair;

for i = 1:length(points)-1
    for j = i+1:length(points)
        if sqrt((points_2(1,i)-points_2(1,j))^2 + (points_2(3,i)-points_2(3,j))^2)...
                < dis_thres || ...
                sqrt((points_2(1,i)-points_2(2,j))^2 + (points_2(3,i)-points_2(3,j))^2)...
                < dis_thres
            dis_thres_count=dis_thres_count+1;
            index_pair(dis_thres_count,1) = i;
            index_pair(dis_thres_count,2) = j;
        end
    end
end
      
%% find all intersect points for all possible lane marker
lineLength = 500;
%threshold initialization
dis_thres = 20;
slope_thres = 0.2;
error_flag = 0;
dul_error_flag = 0;
%segment_index = zeros(length(local_max_pt), length(points_2)); % same num
%of segment ?????


figure;
imshow(I)
% imshow(BW);
hold on;

tic;
for i = 1:length(local_max_pt) %main loop starts here
    
angle(i) = local_max_pt(i,1);
tho(i) = local_max_pt(i,2) - length(R(:,1))/2;

ct_width_2(i) = ct_width + tho(i)*sind(90-angle(i));
ct_height_2(i) = ct_width - tho(i)*sind(angle(i));
angle_41(i) = angle(i) - 90;
x_41(i,1) = ct_width_2(i) + lineLength * cosd(angle_41(i));
y_41(i,1) = ct_height_2(i) - lineLength * sind(angle_41(i));
x_41(i,2) = ct_width_2(i) - lineLength * cosd(angle_41(i));
y_41(i,2) = ct_height_2(i) + lineLength * sind(angle_41(i));

% y = kx+b different coordinate system here
k(i) = (y_41(i,2)-y_41(i,1))/(x_41(i,2)-x_41(i,1));
b(i) = ct_height_2(i)-k(i)*ct_width_2(i);

if(b(i)<= 600 && b(i)>=0)
    %disp('intersect case 1 ..');
    intersec_pt(i,:) = [0 b(i)];
elseif((600-b(i))/k(i)>=0 && (600-b(i))/k(i)<=600)
    %disp('intersect case 2 ..');
    intersec_pt(i,:) = [(600-b(i))/k(i) 600];
elseif(600*k(i)+b(i)>=0 && 600*k(i)+b(i)<=600)
    %disp('intersect case 3 ..');
    intersec_pt(i,:) = [600 600*k(i)+b(i)]; 
end
% to find ONE start point of for each lane from LSD
% need to meet both thres/angle requirements 
for j = 1:length(points_2) 
    LSD_dis(i,j) = sqrt((points_2(1,j)-intersec_pt(i,1))^2+(points_2(3,j)-intersec_pt(i,2))^2); 
end
min_index(i) = find(LSD_dis(i,:) == min(LSD_dis(i,:)));

if(abs(points_2(6,min_index(i)) - k(i)) < slope_thres)
    %disp('start pt slope check pass');
else
    %disp('error, slope check not pass, need to reinitialize start pt...')
    %IF ERROR HERE. more code here to renitialize another correct start
    %point....
end

start_pt(:,i) = [points_2(1,min_index(i)) points_2(3,min_index(i))];

% l21 = line(x_41(i,:),y_41(i,:));
% l21.Color = 'red';
% l21.LineStyle = '-';
% l21.LineWidth = 6;

% scatter(points_2(1,min_index(i)),points_2(3,min_index(i)),'MarkerEdgeColor',[0 1 0],...
%               'MarkerFaceColor',[0 .7 .7],...
%               'LineWidth',8);
          
% conbining those line segment based on dis/angle thres
temp_index = min_index(i);
temp_start_pt_x = points_2(1,min_index(i));
temp_start_pt_y = points_2(3,min_index(i));
temp_end_pt_x = points_2(2,min_index(i));
temp_end_pt_y = points_2(4,min_index(i));
last_seg_min_index = min_index(i);

segment_count = 1;
% error_flag = 0;
segment_index(i,1) = min_index(i);

for n = 1:length(points_2)

for m = 1:length(points_2)
    dist_line_end(i,m) = sqrt((points_2(1,m) - temp_end_pt_x)^2 + (points_2(3,m)...
        - temp_end_pt_y)^2);
end
temp_seg_min_index(i) = find(dist_line_end(i,:) == min(dist_line_end(i,:)));

old_start_pt_x = temp_start_pt_x;
old_start_pt_y = temp_start_pt_y;

temp_start_pt_x = points_2(1,temp_seg_min_index(i));
temp_start_pt_y = points_2(3,temp_seg_min_index(i));

if(temp_seg_min_index(i) == last_seg_min_index)
    dul_error_flag = 1;
    break;
    
elseif(sqrt((temp_start_pt_x - temp_end_pt_x)^2+(temp_start_pt_y - temp_end_pt_y)^2) < dis_thres)
    %disp('line segment connection dis_thres pass')
    segment_count = segment_count + 1;
    segment_index(i,segment_count) = temp_seg_min_index(i);
    last_seg_min_index = temp_seg_min_index(i);
    temp_end_pt_x = points_2(2,temp_seg_min_index(i));
    temp_end_pt_y = points_2(4,temp_seg_min_index(i));
    
else
    %disp('ERROR,line segment connection dis_thres did not pass')
    %segment_index(i,segment_count) = 0;
    error_flag = 1;
    break;
end
%
end
% figure;
hold on;

if(error_flag == 1)
    %disp('DEBUG ERROR_DIS')
    for o = 1:length(segment_index)
    selected_L = line([points_2(1,segment_index(i,o)) points_2(2,segment_index(i,o))],[points_2(3,segment_index(i,o)) points_2(4,segment_index(i,o))]);
    selected_L.Color = 'blue';
    selected_L.LineStyle = '-';
    selected_L.LineWidth = 10;
    %H = annotation('arrow',[.3 .6],[.3 .8]);
    end
    
    continue;
    
elseif(dul_error_flag == 1)
    
    %disp('DEBUG ERROR DULPLICATE')
        % line drawing reference 
%     l11 = line(x_11,y_11);
%     l11.Color = 'red';
%     l11.LineStyle = '-';
%     l11.LineWidth = 6;
%     
    for o = 1:length(segment_index)
    selected_L = line([points_2(1,segment_index(i,o))...
        points_2(2,segment_index(i,o))],[upper_bound + (lower_bound-upper_bound)/600*points_2(3,segment_index(i,o))...
        upper_bound + (lower_bound-upper_bound)/600*points_2(4,segment_index(i,o))]);
    
    selected_L.Color = 'blue';
    selected_L.LineStyle = '-';
    selected_L.LineWidth = 10;
    hold on;
    %H = annotation('arrow',[.3 .6],[.3 .8]);
    end
    
    continue;
end


end
t7 = toc;
disp(['STEP 7 line segment classification and combining:', num2str(t7)]);


toc

