% cstr.m applies offset free linear MPC
% to the linearized and
% nonlinear CSTR.
% See Pannocchia and Rawlings, AIChE J, 2002.
%
% Edits by Michael Risbeck (April 2015).

global F F0 r E k0 DeltaH rhoCp T0 c0 U Tc hs

pkg('load', 'control');

%<<ENDCHUNK>>

% parameters and sizes for the nonlinear system
delta = 1;
n = 3;
m = 2;
nmeas = n;
np = 1;
small = 1e-5; % Small number.

F0 = 0.1; %         m^3/min
T0 = 350; %         K
c0 =  1;  %         kmol/m^3
r  = 0.219; %       m
k0 = 7.2e10; %      min1
E  =  8750; %       K
U  =  54.94; %      kJ/(min m^2 K)
rho = 1e3;   %      kg/m^3
Cp  = 0.239;  %     kJ/(kg K)
DeltaH = -5e4; %    kJ/kmol
Tcs = 300; %        K
hs = 0.659; %       m
ps = 0.1*F0; %      m^3/min

rhoCp = rho*Cp;

%<<ENDCHUNK>>

function df = partial(x)
    global F F0 r E k0 DeltaH rhoCp T0 c0 U Tc
    c = x(1);
    T = x(2);
    h = x(3);
    k = k0*exp(-E/T);
    kprime = k*E/(T^2);
    Fprime = F0/(pi*r^2*h);
    df = [
        -Fprime - k, -kprime*c, ...
        -Fprime*(c0-c)/h, ...
        0, 0, (c0-c)/(pi*r^2*h);
        %
        -k*DeltaH/rhoCp, -Fprime ...
            - kprime*c*DeltaH/(rhoCp) ...
            - 2*U/(r*rhoCp), ...
        -Fprime*(T0-T)/h, ...  
        2*U/(r*rhoCp), 0, (T0-T)/(pi*r^2*h);
        %
        0, 0, 0, 0, -1/(pi*r^2), 1/(pi*r^2)
    ];
endfunction

function rhs = massenbalstst(x)
    global F F0 r E k0 DeltaH rhoCp T0 c0 U Tc hs
    c = x(1);
    T = x(2);
    h = x(3);
    k = k0*exp(-E/T);
    rate = k*c;
    dcdt = F0*(c0-c)/(pi*r^2*h) - rate;
    dTdt = F0*(T0-T)/(pi*r^2*h) - ...
        DeltaH/rhoCp*rate + ...
        2*U/(r*rhoCp)*(Tc-T); 
    % fix the reactor height
    dhdt = h - hs;
    rhs = [dcdt; dTdt; dhdt];
endfunction

function rhs = massenbal(t, x)
    global F F0 r E k0 DeltaH rhoCp T0 c0 U Tc
    c = x(1);
    T = x(2);
    h = x(3);
    k = k0*exp(-E/T);
    rate = k*c;
    dcdt = F0*(c0-c)/(pi*r^2*h) - rate;
    dTdt = F0*(T0-T)/(pi*r^2*h) - ...
      DeltaH/rhoCp*rate + ...
      2*U/(r*rhoCp)*(Tc-T); 
    dhdt = (F0 - F)/(pi*r^2);
    rhs = [dcdt; dTdt; dhdt];
endfunction

%<<ENDCHUNK>>

%% find the steady-state
Fs = F0; F = F0;
Tc = Tcs;
z0 = [c0; Tc; hs];
[z, fval, info] = fsolve(@massenbalstst, z0);
if info ~= 1
    warning('failure to find steady state.')
end
cs = z(1);
Ts = z(2);
hs = z(3);
zs = [cs; Ts; hs];

%<<ENDCHUNK>>

%check the linear model
G = partial(zs);
Ac =  G(:, 1:n);
Bc = G(:, n+1:n+2);
Bpc = G(:,n+3:end);
C = eye(n);
sys = ss(Ac, [Bc, Bpc], C, ...
    zeros(size(C,1),size([Bc, Bpc],2)));
dsys = c2d(sys, delta);
A = dsys.a;
B = dsys.b(:,1:m);
Bp = dsys.b(:,m+1:end);

%<<ENDCHUNK>>

% set up linear controller
Q = diag(1./zs.^2);
R = diag(1./[Tcs; Fs].^2);
[K, P] = dlqr(A, B, Q, R);
K = -K;

%<<ENDCHUNK>>

% Pick whether to use good disturbance model.
useGoodDisturbanceModel = true();

if useGoodDisturbanceModel
    % disturbance model 6; no offset
    nd = 3;
    Bd = zeros(n, nd);
    Bd(:,3) = B(:,2);
    Cd = [1 0 0; 0 0 0; 0 1 0];
else
    % disturbance model with offset
    nd = 2;
    Bd = zeros(n, nd);
    Cd = [1 0; 0 0; 0 1];
end

Aaug = [A, Bd; zeros(nd, n), eye(nd)];
Baug = [B; zeros(nd, m)];
Caug = [C, Cd];
naug = size(Aaug,1);

% detectability test of disturbance model
detec = rank([eye(n+nd) - Aaug; Caug]);
if detec < (n + nd)
  warning ('augmented system is not detectable\n')
end

%<<ENDCHUNK>>

% set up state estimator; use KF
Qw = zeros(naug);
Qw(1:n,1:n) = small*eye(n);
Qw(n+1:end,n+1:end) = small*eye(nd); Qw(end,end)=1.0;
Rv = diag((5e-4*zs).^2);
[L, M, P] = dlqe(Aaug, eye(naug), Caug, Qw, Rv);
Lx = L(1:n,:);
Ld = L(n+1:end,:);

%<<ENDCHUNK>>

ntimes = 50;
x0 = zeros(n, 1);
x = zeros(n, ntimes);
x(:, 1) = x0;
y = zeros(nmeas, ntimes);
u = zeros(m, ntimes);
randn('seed', 0);
v = zeros(nmeas, ntimes);
xhat_ = zeros(n, ntimes);
dhat_ = zeros(nd, ntimes);
xhat = xhat_;
dhat = dhat_;
time = (0:ntimes-1)*delta;
xs = zeros(n, ntimes);
us = zeros(m, ntimes);
ys = zeros(nmeas, ntimes);
etas = zeros(nmeas, ntimes);
options = [];

%<<ENDCHUNK>>

% Disturbance and setpoint.
p = [zeros(np, 10), ps*ones(np, ntimes-10)];
yset = zeros(nmeas,1);

%<<ENDCHUNK>>

% Steady-state target matrices.
H  = [1 0 0; 0 0 1];
Ginv = inv([eye(n)-A, -B; ...
    H*C, zeros(size(H,1), m)]);

%<<ENDCHUNK>>

for i = 1: ntimes
    %% measurement
    y(:,i) = C*x(:,i) + v(:,i);
    
    %% state estimate
    ey = y(:,i) - C*xhat_(:,i) -Cd*dhat_(:,i);
    xhat(:,i) = xhat_(:,i) + Lx*ey;
    dhat(:,i) = dhat_(:,i) + Ld*ey;
    
    % Stop if at last time.
    if i == ntimes
        break
    end
    
    %<<ENDCHUNK>>
    
    %% target selector
    tmodel.p = dhat(:,i);
    
    qs = Ginv*[Bd*dhat(:,i); H*(yset-Cd*dhat(:,i))];
    xss = qs(1:n); 
    uss = qs(n+1:end);
    xs(:,i) = xss;
    us(:,i) = uss;
    ys(:,i) = C*xss + Cd*dhat(:,i);
    
    %<<ENDCHUNK>>
    
    %% control law
    x0 = xhat(:,i) - xs(:,i);
    u(:,i) = K*x0 + us(:,i);

    %<<ENDCHUNK>>

    %% plant evolution
    t = [time(i); mean(time(i:i+1)); time(i+1)];
    z0 = x(:,i) + zs;
    Tc = u(1,i) + Tcs;
    F  = u(2,i) + Fs;
    F0 = p(:,i) + Fs;
    if exist('OCTAVE_VERSION','builtin')
        z = lsode(@(x,t) massenbal(t,x), z0, t);
        tout = t;
    else
        [tout, z] = ode15s(@massenbal, t, ...
            z0, options);
    end
    if sum(tout ~= t)
        warning('integrator failed!')
    end
    x(:,i+1) = z(end,:)' - zs;
    
    %<<ENDCHUNK>>
    
    %% advance state estimates
    xhat_(:,i+1) = A*xhat(:,i) + ...
        Bd*dhat(:,i) + B*u(:,i);
    dhat_(:,i+1) = dhat(:,i);
end
u(:,end) = u(:,end-1); % Repeat for stair plot.

%<<ENDCHUNK>>

% dimensional units
yd = y + kron(ones(1, ntimes), zs);
ud = u + kron(ones(1, ntimes), [Tcs; Fs]);

% *** Plots ***
figure()
axmul = eye(4) + .05*[1,-1,0,0; ...
    -1,1,0,0;0,0,1,-1;0,0,-1,1];
subplot(3,2,1)
plot(time, yd(1,:), '-or')
ylabel('c (kmol/m^3)')
axis(axis()*axmul)
subplot(3,2,3)
plot(time, yd(2,:), '-or')
ylabel('T (K)')
axis(axis()*axmul)
subplot(3,2,5)
plot(time, yd(3,:), '-or')
ylabel('h (m)')
xlabel('Time')
axis(axis()*axmul)
subplot(3,2,2)
stairs(time, ud(1,:), '-r')
ylabel('Tc (K)')
axis(axis()*axmul)
subplot(3,2,4)
stairs(time, ud(2,:), '-r')
ylabel('F (m^3/min)')
xlabel('Time')
axis(axis()*axmul)
print('cstr_octave.pdf','-dpdf','-S720,600')

