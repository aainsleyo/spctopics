
clear;
clc;
close all;

%% PARAMETERS

b = 1;
c = 1;

a_values = [0.1, 0.5];

h = 0.001;
nStep = 20000;


%% 1. EQUILIBRIUM DIAGRAMS: v* versus G

figure;

for k = 1:length(a_values)

    a = a_values(k);

    G_SN = -(a+c)^2/(4*a*c);

    G_values = linspace(G_SN, 5, 500);

    v_plus = zeros(size(G_values));
    v_minus = zeros(size(G_values));

    for j = 1:length(G_values)

        G = G_values(j);

        [eq_plus, eq_minus] = equilibria(a,b,c,G);

        v_plus(j) = eq_plus(2);
        v_minus(j) = eq_minus(2);

    end

    subplot(1,2,k);

    plot(G_values,v_plus,'b','LineWidth',1.5);
    hold on;

    plot(G_values,v_minus,'Color',[0.9 0.5 0.1], ...
        'LineWidth',1.5);

    xline(G_SN,'k--','Saddle-node');

    % Check for Hopf bifurcation

    G_H = -(a+2)/(a+1)^2;

    det_H = (1-2*a-a^2)/(a+1);

    if det_H > 0

        v_H = -a*(a+2)/(a+1);

        plot(G_H,v_H,'ro','MarkerFaceColor','r', ...
            'MarkerSize',8);

        title(sprintf('a = %.1f (Hopf)',a));

        fprintf('a = %.1f: Hopf at G = %.4f\n',a,G_H);

    else

        title(sprintf('a = %.1f (No Hopf)',a));

        fprintf('a = %.1f: No Hopf bifurcation\n',a);

    end

    xlabel('G');
    ylabel('Equilibrium v*');

    legend('Upper branch','Lower branch', ...
        'Saddle-node','Hopf','Location','best');

    grid on;

end

sgtitle('Equilibrium diagrams');

print(gcf,'equilibrium_diagrams.pdf','-dpdf');


%% 2. PHASE-SPACE PORTRAITS

% Case 1: a = 0.1 (Hopf)

phase_portrait(0.1,[-2.0,-1.7355,1.6],b,c,h,nStep);

% Case 2: a = 0.5 (No Hopf)

phase_portrait(0.5,[-1.0,0.0,1.6],b,c,h,nStep);


%% EQUILIBRIUM FUNCTION

function [eq_plus,eq_minus] = equilibria(a,b,c,G)

    D = (a+c)^2 + 4*a*c*G;

    if D < 0
        eq_plus = [NaN,NaN];
        eq_minus = [NaN,NaN];
        return;
    end

    eq_plus = [
        -(a+c)/(2*a) + sqrt(D)/(2*a), ...
        -(a+c)/(2*c) + sqrt(D)/(2*c)
    ];

    eq_minus = [
        -(a+c)/(2*a) - sqrt(D)/(2*a), ...
        -(a+c)/(2*c) - sqrt(D)/(2*c)
    ];

end


%% PHASE-SPACE FUNCTION

function phase_portrait(a,G_values,b,c,h,nStep)

    figure;

    initial_conditions = [
        -8,-1.8;
        -8,-1.0;
        -4,-2.0;
        -4, 0.0;
         0,-2.0;
         0, 1.0;
         3, 0.0
    ];

    for k = 1:length(G_values)

        G = G_values(k);

        p = [a,b,c,G];

        subplot(1,length(G_values),k);

        hold on;

        %% Vector field

        u_grid = linspace(-12,8,25);
        v_grid = linspace(-4,2,25);

        [U,V] = meshgrid(u_grid,v_grid);

        f = -V-U.*(V+1)+G;

        dU = -a*U+b*f+c*V;
        dV = f;

        speed = sqrt(dU.^2+dV.^2);
        speed(speed == 0) = 1;

        quiver(U,V,dU./speed,dV./speed,0.45, ...
            'Color',[0.75,0.75,0.75]);

        %% Trajectories

        for j = 1:size(initial_conditions,1)

            u = initial_conditions(j,:)';

            trajectory = zeros(nStep+1,2);
            trajectory(1,:) = u';

            count = 1;

            for step = 1:nStep

                u = RK2step(u,h,p);

                if any(~isfinite(u)) || max(abs(u)) > 1e3 ...
                        || u(2) < -100
                    break;
                end

                count = count+1;
                trajectory(count,:) = u';

            end

            trajectory = trajectory(1:count,:);

            plot(trajectory(:,1),trajectory(:,2), ...
                'LineWidth',1.2);

            plot(initial_conditions(j,1), ...
                initial_conditions(j,2),'ko','MarkerSize',3);

        end

        %% Equilibrium points and eigenvalues

        [eq_plus,eq_minus] = equilibria(a,b,c,G);

        equilibria_list = [eq_plus;eq_minus];

        for j = 1:2

            eq = equilibria_list(j,:)';

            if any(~isfinite(eq))
                continue;
            end

            J = Jacobian(eq,p);

            eigenvalues = eig(J);

            fprintf(['a = %.2f, G = %.4f, ', ...
                'Equilibrium %d: (%f,%f), Eigenvalues: '], ...
                a,G,j,eq(1),eq(2));

            fprintf('%f ',eigenvalues);
            fprintf('\n');

            if j == 1

                plot(eq(1),eq(2),'r*','MarkerSize',12);

            else

                plot(eq(1),eq(2),'bs','MarkerSize',8);

            end

        end

        xlabel('u');
        ylabel('v');

        title(sprintf('a = %.1f, G = %.4f',a,G));

        xlim([-12,8]);
        ylim([-4,2]);

        grid on;

    end

    sgtitle(sprintf('Phase-space portraits: a = %.1f',a));

    filename = sprintf('phase_portraits_a_%.1f.pdf',a);

    print(gcf,filename,'-dpdf');

end


%% RIGHT-HAND SIDE

function r = rhs(u,p)

    a = p(1);
    b = p(2);
    c = p(3);
    G = p(4);

    f = -u(2)-u(1)*(u(2)+1)+G;

    r = zeros(2,1);

    r(1) = -a*u(1)+b*f+c*u(2);
    r(2) = f;

end


%% JACOBIAN

function J = Jacobian(u,p)

    a = p(1);
    b = p(2);
    c = p(3);

    dfdu = -u(2)-1;
    dfdv = -1-u(1);

    J = [
        -a+b*dfdu, b*dfdv+c;
        dfdu,       dfdv
    ];

end


%% SECOND-ORDER RUNGE-KUTTA

function y2 = RK2step(u,h,p)

    f = rhs(u,p);

    y1 = u+0.5*h*f;

    f = rhs(y1,p);

    y2 = u+h*f;

end
