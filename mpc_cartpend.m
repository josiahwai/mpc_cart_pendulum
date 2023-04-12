% Performs Model Predictive Control on the Cart-Pendulum system from Steve
% Brunton's control bootcamp: 
% 
% https://www.youtube.com/playlist?list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m
% 
% Also see: 
% github.com/bertozzijr/Control_Bootcamp_S_Brunton

clear all; close all; clc

%% User-defined settings
 
plotlevel = 2;   % 0 = no plots, 1=minimal plots, 2=lotsa plots

% Model parameters
pendulum = 'up';      % 'up' or 'down'
m = 1;
M = 5;
L = 2;
g = -10;
d = 1;

% define state space model
if strcmp(pendulum, 'up')
  s = 1;
elseif strcmp(pendulum, 'down')
  s = -1;
end
A = [0 1 0 0;
    0 -d/M -m*g/M 0;
    0 0 0 1;
    0 -s*d/(M*L) -s*(m+M)*g/(M*L) 0];  
B = [0; 1/M; 0; s*1/(M*L)];


% simulation timing
t0 = 0;             % start time
tf = 5;             % end time
ts = .01;           % sample time for mpc
Tlook = 1;          % lookahead time for mpc (should be integer multiple of ts)
Nlook = Tlook/ts;   % number of lookahead steps
tspan = t0:ts:tf;   % timebase for simulation
N = length(tspan);  % number of simulation steps


% control weights  (uses the same weights for both LQR and MPC)
Q = diag([10 1 10 1]);  
R = 0.1;


% discretize system
[Ad,Bd] = c2d(A,B,ts);      


% MPC terminal cost: 
% If using the constrained-LQR formulation, and if all the constraints are 
% inactive throughout the simulation, then the MPC and LQR results are
% identical

% Qf = Q;                    % don't use a special cost for the terminal state 
Qf = idare(Ad,Bd,Q,R);       % constrained-LQR formulation
% Qf = icare(A,B,Q,R) / ts;  % another method for defining constrained-LQR formulation


% initial conditions and targets
if strcmp(pendulum, 'up')

  r = [1; 0; pi; 0];      % the target to track 
  xref = [0; 0; pi; 0];   % the point about which the system is linearized
  x0 = [0; 0; pi+.3; 0];  % the initial state 
  dx0 = x0 - xref;        % the linearized initial state

elseif strcmp(pendulum, 'down')

  r = [4; 0; 0; 0];       % the target to track 
  xref = [0 0 0 0]';      % the point about which the system is linearized
  x0 = [0; 0; 0; 0];      % the initial state 
  dx0 = x0 - xref;        % the linearized initial state

end


% MPC state constraints
% Example: place a wall at -1.2
wall = -1.2;
xmin = [wall -inf -inf -inf]';  
xmax = [ inf  inf inf  inf]';


% MPC input constraints 
umin = -inf; 
umax = inf;  

%% Simulate the system using LQR

K = lqr(A,B,Q,R);

% we are using LQR but also simulating the effect of actuators saturating
uclip = @(u) min(max(u, umin),umax);  

% nonlinear simulation of dynamics
[t,xlqr] = ode45(@(t,x)cartpend(x,m,M,L,g,d,uclip(-K*(x-r))),tspan,x0);

% reconstruct what the control input was
ulqr = uclip(-K*(xlqr'-r));




%% Simulate the system using MPC

% discretize
[Ad,Bd] = c2d(A,B,ts);

% we only have a final target, if you want to do time-dependent tracking 
% would have to define rhat from the value of the target, r, at each step
rhat = repmat(r,Nlook,1);  

xrefhat = repmat(xref, Nlook, 1);

% initialize
uhat = zeros(Nlook,1);
u = 0;
xk = x0;     % state
dxk = dx0;   % linearized state (dxk+1 = A*dxk + B*uk)


% expand cost matrices
Qhat = {};
Rhat = {};
for i = 1:Nlook
  Qhat{i} = Q;
  Rhat{i} = R;
  if i == Nlook
    Qhat{i} = Qf;   % terminal cost
  end
end
Qhat = blkdiag(Qhat{:});
Rhat = blkdiag(Rhat{:});


% form the MPC prediction model: dxhat = E*dxk + F*uhat
nx = size(Ad,1);
nu = size(Bd,2);
E = [];
F  = [];
Apow  = eye(nx);
F_row = zeros(nx, Nlook*nu);
for i = 1:Nlook
  idx = (nu*(i-1)+1):(nu*i);
  F_row = Ad * F_row;
  F_row(:,idx) = Bd;
  F = [F; F_row];

  Apow = Ad*Apow;
  E = [E; Apow];
end

% project state constraints into future
xmax = repmat(xmax, Nlook, 1);
xmin = repmat(xmin, Nlook, 1);
Ix = [eye(Nlook*nx); -eye(Nlook*nx)];
Aineq_x = Ix*F;
bineq_x = [xmax; -xmin] - Ix * (E*dxk + xrefhat); 

% project input constraints into future
Iu = [eye(Nlook*nu); -eye(Nlook*nu)];
Aineq_u = Iu;
bineq_u = [umax*ones(Nlook,1); -umin*ones(Nlook,1)];
  

% quadprog settings
quadprog_opts =  optimset('Display','off');
iplot = 0;
cmap = lines;


% Simulate MPC
for i = 1:N
  
  fprintf('Step %d of %d \n', i, N)
 
  H = F'*Qhat*F + Rhat;
  H = (H+H')/2;
  ft = (dxk'*E' + (xrefhat-rhat)') * Qhat * F;
  f = ft';  
  

  % inequality constraints  
  bineq_x = [xmax; -xmin] - Ix * (E*dxk + xrefhat);
  Aineq = [Aineq_x; Aineq_u];
  bineq = [bineq_x; bineq_u];
  iuse = find(~isinf(bineq));
  bineq = bineq(iuse);
  Aineq = Aineq(iuse,:);
  
  
  % Solve for solution
  % uhat = -inv(H)*f;                                             % unconstrained solution
  uhat = quadprog(H,f,Aineq,bineq,[],[],[],[],[],quadprog_opts);  % constrained solution   
  
  u = uhat(1);  % use only the first control input
  

  % simulate the actual nonlinear system over the ts time interval,
  % applying the control input, u
  t_interval = 0:.001:ts;
  [t,x] = ode45(@(t,x)cartpend(x,m,M,L,g,d,u),t_interval,xk);
  

  xk = x(end,:)';     % current state
  dxk = xk - xref;    % current linearized state
  

  % save all the x and u from this mpc simulation
  xmpc(i,:) = xk;
  umpc(i) = u;
  

  % data saving and plotting
  if mod(i,20) == 1
    iplot = iplot + 1;

    umpc_predicted = uhat;                         % predicted future control inputs
    xmpc_predicted = E*dxk + F*uhat + xrefhat;     % predicted future states
    xmpc_predicted = reshape(xmpc_predicted, nx, []);
    tmpc_predicted = tspan(i) + (ts:ts:Tlook);     % timebase for future predictions

    % save the very first MPC predictions
    if i == 1
      umpc_predicted1 = umpc_predicted;
      xmpc_predicted1 = xmpc_predicted;
      tmpc_predicted1 = tmpc_predicted;
    end

    % plot predictions along the way
    if plotlevel >= 2
      figure(1)
      c = cmap(iplot, :);
      subplot(211)
      hold on
      title('Control input: u', 'fontsize', 16)
      xlabel('Time [s]', 'fontsize', 14)
      scatter(tspan(i), u, 'markerfacecolor', c);
      plot(tmpc_predicted, umpc_predicted, '-', 'color', c)
      if i == 1
        legend('True trajectory', 'MPC-predicted trajectory', ...
          'autoupdate', 'off', 'fontsize', 14);
      end
      subplot(212)
      hold on
      title('Control states: x', 'fontsize', 16)
      xlabel('Time [s]', 'fontsize', 14)
      scatter(tspan(i) * ones(4,1), xk, 'markerfacecolor', c);
      plot(tmpc_predicted, xmpc_predicted, '-', 'color', c)
    end  
  end  
end



%%  Plot results

% Plot timetraces
if plotlevel >= 2
  co = mat2cell(colororder, ones(7,1), 3);
  
  figure
  subplot(211)
  grid on
  title('u', 'fontsize', 18)
  hold on
  title('Control input: u', 'fontsize', 16)
  xlabel('Time [s]', 'fontsize', 14)
  plot(tspan, ulqr, 'color', co{1}, 'linewidth', 1.5)
  plot(tspan, umpc, 'color', co{2}, 'linewidth', 1.5)
  plot(tmpc_predicted1, umpc_predicted1, 'color', co{3}, 'linestyle', '--', 'linewidth', 1.5)
  if ~isinf(umin)
    yline(umin, 'r', 'linewidth', 2)
    yline(umax, 'r', 'linewidth', 2)
  end
  legend('LQR', 'MPC', 'MPC prediction @ t=0', 'fontsize', 16)
  xlim([tspan(1) tspan(end)])
  
  
  subplot(212)
  grid on
  hold on
  title('Control states: x', 'fontsize', 16)
  xlabel('Time [s]', 'fontsize', 14)
  plot(tspan, xlqr, 'color', co{1}, 'linewidth', 1.5)
  plot(tspan, xmpc, 'color', co{2}, 'linewidth', 1.5)
  plot(tmpc_predicted1, xmpc_predicted1, 'color', co{3}, 'linestyle', '--', 'linewidth', 1.5)
  xlim([tspan(1) tspan(end)])
  if ~isinf(wall)
    yline(wall, 'r', 'linewidth', 2)
  end
  xlabel('Time [s]')
end


% Animations
if plotlevel >= 1

  % plot the cart animations
  disp('Drawing MPC animation...')
  figure
  klist = floor(linspace(1, N, 100));
  for k = klist
    drawcartpend_bw(xmpc(k,:),m,M,L,wall);
    title('Model Predictive Control', 'color', 'w', 'fontsize', 20)
    drawnow
  end


  % plot the cart animations
  disp('Drawing LQR animation...')
  figure
  klist = floor(linspace(1, N, 100));
  for k = klist
    drawcartpend_bw(xlqr(k,:),m,M,L,wall);
    title('Linear Quadratic Regulator', 'color', 'w', 'fontsize', 20)
    drawnow
  end
end

disp('done.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IN-FILE FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% nonlinear dynamics for cartpend
function dy = cartpend(y,m,M,L,g,d,u)

  Sy = sin(y(3));
  Cy = cos(y(3));
  D = m*L*L*(M+m*(1-Cy^2));
  
  dy(1,1) = y(2);
  dy(2,1) = (1/D)*(-m^2*L^2*g*Cy*Sy + m*L^2*(m*L*y(4)^2*Sy - d*y(2))) + m*L*L*(1/D)*u;
  dy(3,1) = y(4);
  dy(4,1) = (1/D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*y(4)^2*Sy - d*y(2))) - m*L*Cy*(1/D)*u +.01*randn;
end


% plotting function for cartpend animation
function drawcartpend_bw(y,m,M,L,wall_x)
  x = y(1);
  th = y(3);
  
  % dimensions
  % L = 2;  % pendulum length
  W = 1*sqrt(M/5);  % cart width
  H = .5*sqrt(M/5); % cart height
  wr = .2; % wheel radius
  mr = .3*sqrt(m); % mass radius

  % positions  
  y = wr/2+H/2; % cart vertical position
  w1x = x-.9*W/2;
  w1y = 0;
  w2x = x+.9*W/2-wr;
  w2y = 0;

  px = x + L*sin(th);
  py = y - L*cos(th);

  hold off
  plot([-10 10],[0 0],'w','LineWidth',2)
  hold on
  rectangle('Position',[x-W/2,y-H/2,W,H],'Curvature',.1,'FaceColor',[1 0.1 0.1],'EdgeColor',[1 1 1])
  rectangle('Position',[w1x,w1y,wr,wr],'Curvature',1,'FaceColor',[1 1 1],'EdgeColor',[1 1 1])
  rectangle('Position',[w2x,w2y,wr,wr],'Curvature',1,'FaceColor',[1 1 1],'EdgeColor',[1 1 1])
  
  plot([x px],[y py],'w','LineWidth',2)

  rectangle('Position',[px-mr/2,py-mr/2,mr,mr],'Curvature',1,'FaceColor',[.3 0.3 1],'EdgeColor',[1 1 1])

  % draw the wall
  if exist('wall_x', 'var') && ~isinf(wall_x)
    xline(wall_x - 0.05 - W/2, 'color', 'w', 'Linewidth', 4)
  end

  
  xlim([-5 5]);
  ylim([-2 2.5]);
  set(gca,'Color','k','XColor','w','YColor','w')
  set(gcf,'Position',[10 900 800 400])
  set(gcf,'Color','k')
  set(gcf,'InvertHardcopy','off')     
end













