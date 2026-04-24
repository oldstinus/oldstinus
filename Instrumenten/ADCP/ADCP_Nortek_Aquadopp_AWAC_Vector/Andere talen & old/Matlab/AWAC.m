% Retrieve Currents Data from the AWAC
%
%
awac = open('Herculesa660602_23092015_14112015_p.dat');
% awac = open('a660602.dat');
%
%
% tem_pr = awac.a660602(1:21:end,:);

tem_pr = awac.Herculesa660602_23092015_14112015_p(1:21:end,:);
%
% 
curr = awac.Herculesa660602_23092015_14112015_p;
% curr = awac.a660602;
curr(1:21:end,:) = [ ];
curr(:,11:19) = [ ];
press_aw = tem_pr(:,14);
%
%
index1 = find(curr(:,1) == 1);
cell1 = curr(index1,:);
% cell1(:,11:19) = [ ];
%
%
index2 = find(curr(:,1) == 2);
cell2 = curr(index2,:);
% 
% 
index3 = find(curr(:,1) == 3);
cell3 = curr(index3,:);
% 
% 
index4 = find(curr(:,1) == 4);
cell4 = curr(index4,:);
% 
% 
index5 = find(curr(:,1) == 5);
cell5 = curr(index5,:);
% 
% 
index6 = find(curr(:,1) == 6);
cell6 = curr(index6,:);
%
%
index8 = find(curr(:,1) == 8);
cell8 = curr(index8,:);
%
%
data = find((tem_pr(:,1) == 10) & (tem_pr(:,2) == 14));
%
dir1 = cell1(data,10);
dir2 = cell2(data,10);
dir3 = cell3(data,10);
dir4 = cell4(data,10);
dir5 = cell5(data,10);
dir6 = cell6(data,10);
dir8 = cell8(data,10);
%

%
sea_aw = (press_aw(data)*1.019716) + 1.40;