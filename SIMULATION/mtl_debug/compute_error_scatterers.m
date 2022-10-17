clearvars
close all

% --- path
ppos0 = "/home/laine/cluster/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER/tech_001/tech_001_id_054_FIELD/phantom/plan_pos_53.mat"; 
ppos1 = "/home/laine/cluster/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER/tech_001/tech_001_id_054_FIELD/phantom/plan_pos_54.mat";

pof = '/home/laine/cluster/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER/tech_001/tech_001_id_054_FIELD/phantom/OF_53_54.nii';
pcf = '/home/laine/cluster/PROJECTS_IO/SIMULATION/SEQ_MEIBURGER/tech_001/tech_001_id_054_FIELD/phantom/image_information.mat';

% --- load data
pos0 = load(ppos0);
pos1 = load(ppos1);
of = niftiread(pof);
data = load(pcf);
cf = data.image.CF;

% --- get position
pos_x0 = pos0.Pos_0(:,1);
pos_z0 = pos0.Pos_0(:,3);

pos_x1 = pos1.Pos_1(:,1);
pos_z1 = pos1.Pos_1(:,3);

of_x = of(:,:,1); % flow is saved in pixel displacement
of_x  = of_x(:) * cf;
of_z = of(:,:,3); % flow is saved in pixel displacement
of_z  = of_z(:) * cf;

[height, width, ~] = size(of);
% --- compute error
err_x = of_x + pos_x0 - pos_x1;
err_z = of_z + pos_z0 - pos_z1;

err_x = reshape(err_x, [height, width]);
err_z = reshape(err_z, [height, width]);


% ---
subplot(1,5,1)
% imagesc(pos_x0)
scatter(pos_x0(1:1000:end),pos_z0(1:1000:end), 'filled', "blue")
colorbar
title('scatt at t=0')
subplot(1,5,2)
scatter(pos_x1(1:1000:end),pos_z1(1:1000:end), 'filled', "red")
colorbar
title('scatt at t=1')
subplot(1,5,3)
id = [10, 20, 30, 40, 50, 60, 70];
scatter(pos_x0(id),pos_z0(id), 'filled', "blue")
hold on
scatter(pos_x1(id),pos_z1(id), 'filled', "red")
hold on
h = quiver(pos_x0(id),pos_z0(id),...
           of_x(id),of_z(id), 'off');
hold off
title('displacement field (sample)')
subplot(1,5,4)
imagesc(err_x)
colormap hot
colorbar
title('posx0 + dx - posx1')
subplot(1,5,5)
imagesc(err_z)
colormap hot
colorbar
title('posz0 + dz - posz1')