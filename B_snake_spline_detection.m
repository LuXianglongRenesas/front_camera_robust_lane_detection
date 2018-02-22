%%
close all;
clear all;
clc

%% show the image.
%im = imread('front_straight.png');
%I = imread('front_curved.png');
%I = imread('front_straight.png');
%I = imread('./cordova1/f00013.png');

%%
I = imread('Matt_1.png');
I = imresize(I, [900 600]);

%% Image Crop and section initializations
figure(1)
subplot(1,3,1)
imshow(I);
hold on;
%draw ROI sections 
upper_bound = 250; lower_bound = 850;
line_x = 0:600;
line_y1 = upper_bound; line_y2 = lower_bound;
plot(line_x,line_y1,'-o','MarkerSize',2,'Color',[1 0 0]);
plot(line_x,line_y2,'-o','MarkerSize',2,'Color',[1 0 0]);

I_ROI = imcrop(I,[0 upper_bound 600 lower_bound-upper_bound]);
subplot(1,3,2); I_ROI_resize = imresize(I_ROI, [600 600]);
imshow(I_ROI_resize)

%cut ROI into 7 sections 
sec_wid_1 = 150;sec_wid_2 = 150;sec_wid_3 = 100;
sec_wid_4 = 100;sec_wid_5 = 50;sec_wid_6 = 25;sec_wid_7 = 25;
%draw those sections 
hold on;
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

% All filters start here
GS = rgb2gray(I_ROI_resize);
hsvImage = rgb2hsv(I_ROI_resize);
h = hsvImage(:,:,1);
s = hsvImage(:,:,2);
v = hsvImage(:,:,3);
%whitePixels = h<0.05 & s < 0.18 & v>0.5;
whitePixels = v > 0.7;
%simple color filtering here before sobel
%Bin_img = imbinarize(GS,'0.7');
%Bin_img = GS > 180;
BW = edge(whitePixels,'Sobel');
subplot(1,3,3)
imshow(BW);
imwrite(BW,'temp_BW.png');
points = lsd('temp_BW.png');
hold on;
%apply LSD here
% for i = 1:size(points, 2)
%     plot(points(1:2, i), points(3:4, i), 'LineWidth', points(5, i) / 2, 'Color', [0, 1, 0]);
% end

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

figure(3)
hold on;
axes('pos',[0 0 1 1]);
axis([0 10 0 10])
%lines format [x1 x2 y1 y2 line_width] in matrix points 
for i = 1:length(points_2)
H = annotation('arrow',[points_2(1,i)/1000 points_2(2,i)/1000],...
    [(600-points_2(3,i))/1000 (600-points_2(4,i))/1000]);
%H = annotation('arrow',[.3 .6],[.3 .8]);
end

%%
num_uni_seg = 0;
dul_seg_dis_thres = 10;
points_4 = points_2;

%try to combing similar line segments
for i = 1:length(points_2)-1
    for j = i+1:length(points_2)
        %check distance threshold
        if (abs(points_2(1,i) - points_2(1,j)) < dul_seg_dis_thres && ...
           abs(points_2(2,i) - points_2(2,j)) < dul_seg_dis_thres && ...
           abs(points_2(3,i) - points_2(3,j)) < dul_seg_dis_thres && ...
           abs(points_2(4,i) - points_2(4,j)) < dul_seg_dis_thres)
       
       num_uni_seg = num_uni_seg + 1;
       points_3(:,num_uni_seg) = points_2(:,j);
       dul_index(num_uni_seg) = j;
        end
     
    end
end
dul_index = unique(dul_index);
points_4(:,dul_index) = [];


%
figure;
for i = 1:length(points_4)
H_3 = annotation('arrow',[points_4(1,i)/1000 points_4(2,i)/1000],...
    [(600-points_4(3,i))/1000 (600-points_4(4,i))/1000]);
%H = annotation('arrow',[.3 .6],[.3 .8]);
end

%% Radon transform (roughly detect how many lines are there)
theta = 0:180;
figure(4)
[R,xp] = radon(BW,theta);
imagesc(theta,xp,R);
title('R_{\theta} (X\prime)');
xlabel('\theta (degrees)');
ylabel('X\prime');
set(gca,'XTick',0:20:180);
colormap(hot);
colorbar

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
lineLength = 1000;
max_segment_num = 15;
segment_index = zeros(length(local_max_pt),max_segment_num);
%threshold initialization
dis_thres = 50;
slope_thres = 0.2;
error_flag = 0;
max_segment_num_flag = 0;
dul_error_flag = 0;
%segment_index = zeros(length(local_max_pt), length(points_2)); % same num
%of segment ?????
figure(5);
%subplot(2,2,1);
imshow(I)
% imshow(BW);
hold on;

for i = 1:length(local_max_pt) %main loop starts here
    
error_flag = 0;
max_segment_num_flag = 0;
dul_error_flag = 0;
end_of_this_line_flag = 0 ;
    
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
    disp('intersect case 1 ..');
    intersec_pt(i,:) = [0 b(i)];
elseif((600-b(i))/k(i)>=0 && (600-b(i))/k(i)<=600)
    disp('intersect case 2 ..');
    intersec_pt(i,:) = [(600-b(i))/k(i) 600];
elseif(600*k(i)+b(i)>=0 && 600*k(i)+b(i)<=600)
    disp('intersect case 3 ..');
    intersec_pt(i,:) = [600 600*k(i)+b(i)]; 
end
% to find ONE start point of for each lane from LSD
% need to meet both thres/angle requirements 
for j = 1:length(points_2) 
    LSD_dis(i,j) = sqrt((points_2(1,j)-intersec_pt(i,1))^2+(points_2(3,j)-intersec_pt(i,2))^2); 
end
min_index(i) = find(LSD_dis(i,:) == min(LSD_dis(i,:)));

if(abs(points_2(6,min_index(i)) - k(i)) < slope_thres)
    disp('start pt slope check pass');
else
    disp('error, slope check not pass, need to reinitialize start pt...')
    %IF ERROR HERE. more code here to renitialize another correct start
    %point....
end

start_pt(:,i) = [points_2(1,min_index(i)) points_2(3,min_index(i))];

% l21 = line(x_41(i,:),y_41(i,:));
% l21.Color = 'red';
% l21.LineStyle = '-';
% l21.LineWidth = 6;
% 
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

%for n = 1:length(points_2)
for n = 1:max_segment_num
%     if(last_seg_min_index == n)
%         continue;
%     end

for m = 1:length(points_2)
    
    dist_line_end(i,m) = sqrt((points_2(1,m) - temp_end_pt_x)^2 + (points_2(3,m)...
        - temp_end_pt_y)^2);
end

% temp_seg_min_index(i) = find(dist_line_end(i,:) == min(dist_line_end(i,:)));

%if(n == 1)
    temp_seg_min_index(i) = find(dist_line_end(i,:) == min(dist_line_end(i,:)));
    
if(temp_seg_min_index(i) == last_seg_min_index)
    unique_dist_line_end = unique(dist_line_end(i,:));
    temp_seg_min_index(i) = find(dist_line_end(i,:) == unique_dist_line_end(2));
    
else
    temp_seg_min_index(i) = find(dist_line_end(i,:) == min(dist_line_end(i,:)));
end

if(ismember(temp_seg_min_index(i),segment_index(i,:)) == 1)
    dul_error_flag = 1;
end

old_start_pt_x = temp_start_pt_x;
old_start_pt_y = temp_start_pt_y;

temp_start_pt_x = points_2(1,temp_seg_min_index(i));
temp_start_pt_y = points_2(3,temp_seg_min_index(i));

% if(temp_seg_min_index(i) == last_seg_min_index)
%     dul_error_flag = 1;
%     %break;
if(n == max_segment_num)
    max_segment_num_flag = 1;
    break;
end

if(sqrt((temp_start_pt_x - temp_end_pt_x)^2+(temp_start_pt_y - temp_end_pt_y)^2) < dis_thres)
    disp('line segment connection dis_thres pass')
    
    
else
    disp('ERROR,line segment connection dis_thres did not pass')
    %segment_index(i,segment_count) = 0;
    error_flag = 1;
    break;
end

if(error_flag == 1 || max_segment_num_flag==1 || dul_error_flag == 1)

    hold on;
    disp('DEBUG ERROR_DIS')
    end_of_this_line_flag = 1;
    
    for o = 1:length(segment_index)
        if(segment_index(i,o)==0)
            break;
        end
    selected_L = line([points_2(1,segment_index(i,o))...
        points_2(2,segment_index(i,o))],[upper_bound + (lower_bound-upper_bound)/600*points_2(3,segment_index(i,o))...
        upper_bound + (lower_bound-upper_bound)/600*points_2(4,segment_index(i,o))]);
    
    selected_L.Color = 'blue';
    selected_L.LineStyle = '-';
    selected_L.LineWidth = 10;
    %H = annotation('arrow',[.3 .6],[.3 .8]);
    end
    
    break;
    
else
    segment_count = segment_count + 1;
    segment_index(i,segment_count) = temp_seg_min_index(i);
    last_seg_min_index = temp_seg_min_index(i);
    temp_end_pt_x = points_2(2,temp_seg_min_index(i));
    temp_end_pt_y = points_2(4,temp_seg_min_index(i));
    
end

% if(end_of_this_line_flag)
%     break;
% end

end

end
%% Polynomial fitting and visualization
%calculate mid point of all segments
mid_x = zeros(length(local_max_pt),max_segment_num);
mid_y = zeros(length(local_max_pt),max_segment_num);
polyfit_para = zeros(length(local_max_pt),3);

%sample_x = zeros(length(local_max_pt),max_segment_num);

for i = 1:length(local_max_pt)
    for o = 1:max_segment_num
        if(segment_index(i,o) == 0)
            break;
        else
            mid_x(i,o) = (points_2(1,segment_index(i,o)) + points_2(2,segment_index(i,o)))/2;
            mid_y(i,o) = 600 - (points_2(3,segment_index(i,o)) + points_2(4,segment_index(i,o)))/2;
        end
    end
    polyfit_para(i,:) = polyfit(nonzeros(mid_y(i,:)),nonzeros(mid_x(i,:)),2);
end

%regenerate points for visualization
figure(6)
imshow(I_ROI_resize)
hold on;

for i = 1:length(local_max_pt) %% regeneration of points problem here need to be fixed.
%sample_y(i,:) = 100:100:600;
sample_y = mid_y;
%sample_x(i,:) =  polyfit_para(i,1)*sample_y(i,:).^2 +  polyfit_para(i,2)*sample_y(i,:) +  polyfit_para(i,3);
sample_x(i,1:length(nonzeros(sample_y(i,:)))) = polyval(polyfit_para(i,:),nonzeros(sample_y(i,:)))';

poly_L = line(nonzeros(sample_x(i,1:length(nonzeros(sample_y(i,:))))),600-nonzeros(sample_y(i,:)));

poly_L.Color = 'green';
poly_L.LineStyle = '-';
poly_L.LineWidth = 5;

end

%% More studies on radon lines and box based line segment classification 
figure(7)
% set(hFig, 'Position', [0, 0, 600, 600])
%hold on; 
pixel_thres = 10;
imshow(I_ROI_resize)
for i = 1:length(local_max_pt) 
l21 = line(x_41(i,:),y_41(i,:));
l21.Color = 'red';
l21.LineStyle = '-';
l21.LineWidth = 5;
%draw supportive lines
angle_corrected(i) = angle_41(i);
if (angle_corrected(i)<=0)
    sup_ct_pt_left(i,:) = [ct_width_2(i) - abs(pixel_thres*sind(angle_corrected(i))) ct_height_2(i) + abs(pixel_thres*cosd(angle_corrected(i)))];
    sup_ct_pt_right(i,:) = [ct_width_2(i) + abs(pixel_thres*sind(angle_corrected(i))) ct_height_2(i) - abs(pixel_thres*cosd(angle_corrected(i)))];
else
    sup_ct_pt_left(i,:) = [ct_width_2(i) - abs(pixel_thres*sind(angle_corrected(i))) ct_height_2(i) + abs(pixel_thres*cosd(angle_corrected(i)))];
    sup_ct_pt_right(i,:) = [ct_width_2(i) + abs(pixel_thres*sind(angle_corrected(i))) ct_height_2(i) - abs(pixel_thres*cosd(angle_corrected(i)))];
end

% draw lines
x_sup_left(i,1) = sup_ct_pt_left(i,1) + lineLength * cosd(angle_41(i));
y_sup_left(i,1) = sup_ct_pt_left(i,2) - lineLength * sind(angle_41(i));
x_sup_left(i,2) = sup_ct_pt_left(i,1) - lineLength * cosd(angle_41(i));
y_sup_left(i,2) = sup_ct_pt_left(i,2) + lineLength * sind(angle_41(i));

x_sup_right(i,1) = sup_ct_pt_right(i,1) + lineLength * cosd(angle_41(i));
y_sup_right(i,1) = sup_ct_pt_right(i,2) - lineLength * sind(angle_41(i));
x_sup_right(i,2) = sup_ct_pt_right(i,1) - lineLength * cosd(angle_41(i));
y_sup_right(i,2) = sup_ct_pt_right(i,2) + lineLength * sind(angle_41(i));

l_sup_left = line(x_sup_left(i,:),y_sup_left(i,:));
l_sup_left.Color = rand(1,3);
l_sup_left.LineStyle = '--';
l_sup_left.LineWidth = 1;

l_sup_right = line(x_sup_right(i,:),y_sup_right(i,:));
l_sup_right.Color = l_sup_left.Color;
l_sup_right.LineStyle = '--';
l_sup_right.LineWidth = 1;

% scatter(points_2(1,min_index(i)),points_2(3,min_index(i)),'MarkerEdgeColor',[0 1 0],...
%               'MarkerFaceColor',[0 .7 .7],...
%               'LineWidth',8);
%           
end

%% adaptive offset box generation to limit line segment from selection
%Put line segments into each boxes

num_lanes = local_max_pt;
%slope of all lines
for i = 1:length(num_lanes) 
    slope_k(i) = tand(angle_corrected(i));
    %to get b , b = -y-kx , y = -(kx+b) , x= (-y-b)/k
    b_left(i) = -sup_ct_pt_left(i,2) - slope_k(i)*sup_ct_pt_left(i,1);
    b_right(i) = -sup_ct_pt_right(i,2) - slope_k(i)*sup_ct_pt_right(i,1);
    cnt_loop = 0;
    %check bounding boxes for all possible lines
    for j = 1:length(points_4)
%         y_bound_left_1(i,j) = -(slope_k(i)*points_4(1,j) + b_left(i));
%         y_bound_left_2(i,j) = -(slope_k(i)*points_4(2,j) + b_left(i));
%         y_bound_right_1(i,j) = -(slope_k(i)*points_4(1,j) + b_right(i));
%         y_bound_right_2(i,j) = -(slope_k(i)*points_4(2,j) + b_right(i));
%         
        x_bound_left_1(i,j) = (- points_4(3,j) - b_left(i))/slope_k(i);
        x_bound_left_2(i,j) = (- points_4(4,j) - b_left(i))/slope_k(i);
        x_bound_right_1(i,j) = (- points_4(3,j) - b_right(i))/slope_k(i);
        x_bound_right_2(i,j) = (- points_4(4,j) - b_right(i))/slope_k(i);
      
    
    if( x_bound_left_1(i,j) < points_4(1,j) && points_4(1,j) < x_bound_right_1(i,j) && ...
            x_bound_left_2(i,j) < points_4(2,j) && points_4(2,j) < x_bound_right_2(i,j))
        
        cnt_loop = cnt_loop + 1;
        hit_cnt(i,cnt_loop) = j;
        
    end
    end
    
    figure
    for m = 1:length(nonzeros(hit_cnt(i,:)))
    H_calssified = annotation('arrow',[points_4(1,hit_cnt(i,m))/1000 points_4(2,hit_cnt(i,m))/1000],...
        [(600-points_4(3,hit_cnt(i,m)))/1000 (600-points_4(4,hit_cnt(i,m)))/1000]);
    %H = annotation('arrow',[.3 .6],[.3 .8]);
    end
    

end







