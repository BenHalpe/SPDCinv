%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PDC, with HG01 crystal
% Units in MKS
% 
% pump wave is K2w: here is k1. resulting idler is E3
% Signal wave E2 is a small gaussian with waist omega02 
%
% based on Noa's code
% Sivan Trajtenberg-Mills, Jan. 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear all; close all; pack;

c       = 2.99792458e8;%in meter/sec 
d33     = 16.9e-12;%16.9e-12;%16.9e-12;% in pico-meter/Volt. KTP]
eps0    = 8.854187817e-12; % the vacuum permittivity, in Farad/meter.
I       = @(A,n) 2.*n.*eps0.*c.*abs(A).^2;  
h_bar   = 1.054571800e-34; % Units are m^2 kg / s, taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck

dz=10e-6; dx=2e-6; dy=2e-6; % this was 0.1 um X 0.5 um X 0.5 um
MaxX=120e-6; x=-MaxX:dx:MaxX;
MaxY=120e-6; y=-MaxY:dy:MaxY;
MaxZ=1e-4;
[X,Y] = meshgrid(x,y);
Power2D = @(A,n) sum(sum(I(A,n)))*dx*dy;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Interacting Wavelengths%
lambda_p=405e-9;%1064e-9;%3000e-9;
lambda_s = 2*lambda_p;
lambda_i = 8.1e-7;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


T=50; %temperature, celsius

omega0_p= 40e-6; %pump waist

omega0_i=95e-6; %idler waist  - used if idler is input
omega0_i=95e-6;
omega0_s=100e-6; %signal waist - used if signal is input



PumpPower = 1e-3; %Alphalas: Peak Power 1 MW = 0.1 W * 10 ms / 1 ns


%pump wave
n_p = 1.8406305313110352;
w_p = 4650991524219391.0;
k_p= 28555612.0;
b_p = omega0_p^2*k_p;

%idler wave
n_i = 1.8443180322647095;
w_i = 2325495762109695.0;
k_i= 14306410.0;
b_i = omega0_i^2*k_i;

%signal wave
n_s = 1.7562085390090942;
w_s = 2325495762109695.5;
k_s =13622943.0;
b_s = omega0_s^2*k_s;

delta_k= k_p - k_s - k_i;
Lambda=abs(2*pi/delta_k);


Z= -MaxZ/2:dz:MaxZ/2 ;
%crystal parameters 
Poling_period=delta_k;
PP = cos(abs(delta_k) * Z);
PP=sign(PP);
E0=@(P,n,W0) sqrt(P/(2*n*c*eps0*pi*W0^2)) ;
kappa_i= 2*1i*w_i^2*d33/(k_i*c^2);
kappa_s= 2*1i*w_s^2*d33/(k_s*c^2);
    
    
PumpOffsetX = 0;
FocusZ = 0;

E_i_out=zeros(size(X));
E_i_vac=zeros(size(X));
E_s_out=zeros(size(X));
E_s_vac=zeros(size(X));
E_p=zeros(size(X));

            z_tag = Z(1);
            xi_i=2*(z_tag - FocusZ)./b_i;
            tau_i=1./(1+1i*xi_i);

            xi_p=2*(z_tag - FocusZ)./b_p;
            tau_p=1./(1+1i*xi_p);

            xi_s=2*(z_tag - FocusZ)./b_s;
            tau_s=1./(1+1i*xi_s);

            E_p =(E0(PumpPower,n_p,omega0_p)*tau_p)*exp(-(((X-PumpOffsetX).^2./(omega0_p)^2+(Y).^2./(omega0_p)^2).*tau_p)).*exp(1i*k_p*(z_tag - FocusZ));
            E_i_vac = E_p;
            E_s_vac = E_p;
            

        %%
        for n=1:length(Z)

            disp([num2str(100*n/length(Z)),'% gone!']);
            z_tag=Z(n);
            z_tag2=z_tag-Z(1);
% 
%             xi_i=2*(z_tag - FocusZ)./b_i;
%             tau_i=1./(1+1i*xi_i);
% 
%             xi_p=2*(z_tag - FocusZ)./b_p;
%             tau_p=1./(1+1i*xi_p);
% 
%             xi_s=2*(z_tag - FocusZ)./b_s;
%             tau_s=1./(1+1i*xi_s);
% 
%             E_p =(E0(PumpPower,n_p,omega0_p)*tau_p)*exp(-(((X-PumpOffsetX).^2./(omega0_p)^2+(Y).^2./(omega0_p)^2).*tau_p)).*exp(1i*k_p*(z_tag - FocusZ));
%             if n==1
%              E_i_vac =(E0(1,n_i,omega0_p)*tau_p)*exp(-(((X-PumpOffsetX).^2./(omega0_p)^2+(Y).^2./(omega0_p)^2).*tau_p)).*exp(1i*k_i*(z_tag - FocusZ));
%              E_s_vac =(E0(1,n_s,omega0_p)*tau_p)*exp(-(((X-PumpOffsetX).^2./(omega0_p)^2+(Y).^2./(omega0_p)^2).*tau_p)).*exp(1i*k_s*(z_tag - FocusZ));
%             end

            %generate the crystal slab at this Z
            PP_xy=ones(length(X),width(X),1)*PP(:,n)';
            
            
            E_p = propagate3(E_p,x,y,k_p,dz).*exp(-1i*k_p*Z(n));
            %Non-linear equations:
            dEs_out_dz=kappa_s.*PP_xy.*E_p.*conj(E_i_vac);%*exp(1i*delta_k*z_tag);
            dEs_vac_dz=kappa_s.*PP_xy.*E_p.*conj(E_i_out);%*exp(1i*delta_k*z_tag);
            E_s_out=E_s_out+dEs_out_dz*dz;
            E_s_vac=E_s_vac+dEs_vac_dz*dz;

            %Add the non-linear part
            dEi_out_dz=kappa_i.*PP_xy.*E_p.*conj(E_s_vac);%*exp(-1i*delta_k*z_tag);
            dEi_vac_dz=kappa_i.*PP_xy.*E_p.*conj(E_s_out);%*exp(-1i*delta_k*z_tag);
            E_i_out=E_i_out+dEi_out_dz*dz; % update  
            E_i_vac=E_i_vac+dEi_vac_dz*dz; 

            %Propagate
            E_i_out=propagate3(E_i_out, x, y, k_i, dz); E_i_out=E_i_out.*exp(-1i*k_i*dz);
            E_i_vac=propagate3(E_i_vac, x, y, k_i, dz); E_i_vac=E_i_vac.*exp(-1i*k_i*dz);
            E_s_out=propagate3(E_s_out, x, y, k_s, dz); E_s_out=E_s_out.*exp(-1i*k_s*dz);
            E_s_vac=propagate3(E_s_vac, x, y, k_s, dz); E_s_vac=E_s_vac.*exp(-1i*k_s*dz);
            
            % check solution
            dd_dxx = @(E) (E(3:end,2:end-1)+E(1:end-2,2:end-1)-2*E(2:end-1,2:end-1))/dx^2;
            dd_dyy = @(E) (E(2:end-1,3:end)+E(2:end-1,1:end-2)-2*E(2:end-1,2:end-1))/dy^2;
            trans_laplasian=  @(E) (dd_dxx(E)+dd_dyy(E));
            f = @(dE1_dz,E1,k1,kapa1,E2) (1i*dE1_dz(2:end-1,2:end-1) + trans_laplasian(E1)/(2*k1) - kapa1*PP_xy(2:end-1,2:end-1).*E_p(2:end-1,2:end-1)*exp(-1i*delta_k*z_tag).*conj(E2(2:end-1,2:end-1)))./dE1_dz(2:end-1,2:end-1);
            
            m1 = mean(mean(abs(f(dEi_out_dz,E_i_out,k_i,kappa_i,E_s_vac))^2));
            m2 = mean(mean(abs(f(dEi_vac_dz,E_i_vac,k_i,kappa_i,E_s_out))^2));
            m3 = mean(mean(abs(f(dEs_out_dz,E_s_out,k_s,kappa_s,E_i_vac))^2));
            m4 = mean(mean(abs(f(dEs_vac_dz,E_s_vac,k_s,kappa_s,E_i_out))^2));
            %disp ([m1,m2,m3,m4])
            disp([log(m1)/log(10),log(m2)/log(10),log(m3)/log(10),log(m4)/log(10)]);
        
            
            
        end

% figure; imagesc(I(E_p,n_p));
% figure; imagesc(I(E_i_out,n_i));
% figure; imagesc(I(E_i_vac,n_i));
% figure; imagesc(I(E_s_out,n_s));
% figure; imagesc(I(E_s_vac,n_s));

