function [dst_mask_path, src_mask_path] = create_masks(src_img_path, dst_img_path, src_x, src_y, dst_x, dst_y, width, height)
%CREATE_MASKS Creates masks for source and destination images to be applied
%in start_week2 for Poisson editing

src_img = imread(src_img_path);
dst_img = imread(dst_img_path);

[src_m, src_n, nC] = size(src_img);
src_mask = zeros(src_m, src_n);
[dst_m, dst_n, nC] = size(dst_img);
dst_mask = zeros(dst_m, dst_n);

src_mask(src_y:src_y+height, src_x:src_x+width) = 1;
dst_mask(dst_y:dst_y+height, dst_x:dst_x+width) = 1;

[src_folder, src_filename, ~] = fileparts(src_img_path);
[dst_folder, dst_filename, ~] = fileparts(dst_img_path);
src_mask_path = fullfile(src_folder, sprintf('%s_mask.png', src_filename));
dst_mask_path = fullfile(dst_folder, sprintf('%s_mask.png', dst_filename));

imwrite(logical(src_mask), src_mask_path);
imwrite(logical(dst_mask), dst_mask_path);

end

