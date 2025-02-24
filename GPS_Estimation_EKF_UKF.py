import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  1) Sistem Parametreleri ve Yardımcı Fonksiyonlar
# =============================================================================

# Araç parametreleri (Örnek)
# m       = 668.0
# wheel_b = 1.765
# track   = 1.450

# Örnekleme zaman adımı ve simülasyon süresi
dt      = 0.1         # [s]
sim_time= 150.0       # [s]

# Kovaryans matrisleri
# Durum gürültüsü kovaryansı Q
Q = np.diag([0.1, 0.1, 0.01, 0.01, 0.01])**2

# Ölçüm gürültüsü kovaryansı R
# [ Xg, Yg, Ve, Vn ] ölçülüyor varsayıyoruz
R = np.diag([0.5, 0.5, 0.2, 0.2])**2

def wrap_angle(angle):
    """Açıyı -pi ile +pi arasında tutmak için sarmalama."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

# =============================================================================
#  2) Kinematik Model (State Transition) ve Ölçüm Modeli
# =============================================================================

def f(x, u, dt):
    """
    Sürekli zamanlı modelin Euler ayrıklaştırması.
    x = [Vx, Vy, psi, Xg, Yg]
    u = [ax, ay, w]  -> IMU'dan gelen ivmeler ve yaw_rate
    """
    Vx, Vy, psi, Xg, Yg = x
    ax, ay, w = u

    Vx_next  = Vx + dt * (Vy*w + ax)
    Vy_next  = Vy + dt * (-Vx*w + ay)
    psi_next = wrap_angle(psi + dt * w)
    Xg_next  = Xg + dt * (Vx*np.cos(psi) - Vy*np.sin(psi))
    Yg_next  = Yg + dt * (Vx*np.sin(psi) + Vy*np.cos(psi))
    
    return np.array([Vx_next, Vy_next, psi_next, Xg_next, Yg_next])


def h(x):
    """
    Ölçüm modeli:
      - GPS konumu [Xg, Yg]
      - ENU hız bileşenleri [Ve, Vn]
    """
    Vx, Vy, psi, Xg, Yg = x
    Ve = Vx*np.cos(psi) - Vy*np.sin(psi)
    Vn = Vx*np.sin(psi) + Vy*np.cos(psi)
    return np.array([Xg, Yg, Ve, Vn])

# =============================================================================
#  3) EKF Yardımcı Fonksiyonlar (Jacobian vb.)
# =============================================================================

def jacobian_F(x, u, dt):
    """
    f(x,u) fonksiyonu için Jacobian (EKF'de gerekli).
    x = [Vx, Vy, psi, Xg, Yg], u = [ax, ay, w]
    """
    Vx, Vy, psi, Xg, Yg = x
    ax, ay, w = u

    dfdx = np.zeros((5,5))

    #  d(Vx_next)/dVx = 1
    #  d(Vx_next)/dVy = dt*w
    dfdx[0,0] = 1.0
    dfdx[0,1] = dt * w

    #  d(Vy_next)/dVx = -dt*w
    #  d(Vy_next)/dVy = 1
    dfdx[1,0] = -dt * w
    dfdx[1,1] = 1.0

    #  d(psi_next)/dpsi = 1
    dfdx[2,2] = 1.0

    #  d(Xg_next)/dVx = dt*cos(psi)
    #  d(Xg_next)/dVy = -dt*sin(psi)
    #  d(Xg_next)/dpsi= dt*( -Vx sin(psi) - Vy cos(psi) )
    dfdx[3,0] = dt * np.cos(psi)
    dfdx[3,1] = -dt * np.sin(psi)
    dfdx[3,2] = dt * (-Vx*np.sin(psi) - Vy*np.cos(psi))

    #  d(Yg_next)/dVx = dt*sin(psi)
    #  d(Yg_next)/dVy = dt*cos(psi)
    #  d(Yg_next)/dpsi= dt*( Vx cos(psi) - Vy sin(psi) )
    dfdx[4,0] = dt * np.sin(psi)
    dfdx[4,1] = dt * np.cos(psi)
    dfdx[4,2] = dt * (Vx*np.cos(psi) - Vy*np.sin(psi))

    return dfdx

def jacobian_H(x):
    """
    Ölçüm fonksiyonu h(x) için Jacobian (4x5)
      h(x) = [Xg, Yg, Vx cos(psi)-Vy sin(psi), Vx sin(psi)+Vy cos(psi)]
    """
    Vx, Vy, psi, Xg, Yg = x
    dhdx = np.zeros((4,5))

    # dh/dXg = [1, 0, 0, 0, 0] -> 1,0,0,0,0
    dhdx[0,3] = 1.0   # dXg/dXg
    dhdx[1,4] = 1.0   # dYg/dYg

    # Ve = Vx cos(psi) - Vy sin(psi)
    dhdx[2,0] = np.cos(psi)
    dhdx[2,1] = -np.sin(psi)
    dhdx[2,2] = -Vx*np.sin(psi) - Vy*np.cos(psi)

    # Vn = Vx sin(psi) + Vy cos(psi)
    dhdx[3,0] = np.sin(psi)
    dhdx[3,1] = np.cos(psi)
    dhdx[3,2] =  Vx*np.cos(psi) - Vy*np.sin(psi)

    return dhdx

def ekf_predict(x_est, P_est, u):
    x_pred = f(x_est, u, dt)
    F_k = jacobian_F(x_est, u, dt)
    P_pred = F_k @ P_est @ F_k.T + Q
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z_meas):
    H_k = jacobian_H(x_pred)
    z_pred = h(x_pred)

    S = H_k @ P_pred @ H_k.T + R
    K = P_pred @ H_k.T @ np.linalg.inv(S)

    y = z_meas - z_pred
    x_upd = x_pred + K @ y
    # yaw sarmala
    x_upd[2] = wrap_angle(x_upd[2])

    I = np.eye(len(x_pred))
    P_upd = (I - K @ H_k) @ P_pred
    return x_upd, P_upd

# =============================================================================
#  4) UKF Yardımcı Fonksiyonlar
# =============================================================================

def ukf_sigma_points(x, P, alpha=1e-3, beta=2, kappa=0):
    """
    Sigma noktalarını hesaplar. 2n+1 tane nokta döner.
    """
    n = len(x)
    lam = alpha**2*(n+kappa) - n
    # Cholesky
    U = np.linalg.cholesky((n+lam)*P)

    sigmas = np.zeros((2*n+1, n))
    sigmas[0] = x
    for i in range(n):
        sigmas[i+1]     = x + U[i]
        sigmas[n+i+1]   = x - U[i]
    return sigmas

def ukf_weights(n, alpha=1e-3, beta=2, kappa=0):
    lam = alpha**2*(n+kappa) - n
    Wm = np.zeros(2*n+1)  # mean weights
    Wc = np.zeros(2*n+1)  # cov weights

    Wm[0] = lam/(n+lam)
    Wc[0] = lam/(n+lam) + (1 - alpha**2 + beta)

    for i in range(1, 2*n+1):
        Wm[i] = 1.0/(2*(n+lam))
        Wc[i] = 1.0/(2*(n+lam))
    return Wm, Wc

def ukf_predict(x_est, P_est, u, alpha=1e-3, beta=2, kappa=0):
    n = len(x_est)
    sigmas = ukf_sigma_points(x_est, P_est, alpha, beta, kappa)
    Wm, Wc = ukf_weights(n, alpha, beta, kappa)

    sigmas_pred = np.zeros_like(sigmas)
    for i in range(sigmas.shape[0]):
        sigmas_pred[i] = f(sigmas[i], u, dt)

    # Ortalamayı bul
    x_pred = np.zeros(n)
    for i in range(sigmas_pred.shape[0]):
        x_pred += Wm[i]*sigmas_pred[i]
    x_pred[2] = wrap_angle(x_pred[2])

    # Kovaryansı bul
    P_pred = np.zeros((n,n))
    for i in range(sigmas_pred.shape[0]):
        diff = sigmas_pred[i] - x_pred
        diff[2] = wrap_angle(diff[2])
        P_pred += Wc[i]*np.outer(diff, diff)
    P_pred += Q

    return x_pred, P_pred, sigmas_pred, Wm, Wc

def ukf_update(x_pred, P_pred, sigmas_pred, z_meas, Wm, Wc):
    n = len(x_pred)
    m = len(z_meas)

    # Ölçüm modeli sigma noktalarına uygula
    Zsig = np.zeros((sigmas_pred.shape[0], m))
    for i in range(sigmas_pred.shape[0]):
        Zsig[i] = h(sigmas_pred[i])

    # Z ortalaması
    z_pred = np.zeros(m)
    for i in range(Zsig.shape[0]):
        z_pred += Wm[i]*Zsig[i]

    # Inovasyon kovaryansı
    S = np.zeros((m,m))
    for i in range(Zsig.shape[0]):
        diff_z = Zsig[i] - z_pred
        S += Wc[i]*np.outer(diff_z, diff_z)
    S += R

    # Çapraz kovaryans
    Pxz = np.zeros((n,m))
    for i in range(sigmas_pred.shape[0]):
        diff_x = sigmas_pred[i] - x_pred
        diff_x[2] = wrap_angle(diff_x[2])
        diff_z = Zsig[i] - z_pred
        Pxz += Wc[i]*np.outer(diff_x, diff_z)

    # Kazanç
    K = Pxz @ np.linalg.inv(S)

    # Güncelleme
    y = z_meas - z_pred
    x_upd = x_pred + K @ y
    x_upd[2] = wrap_angle(x_upd[2])
    P_upd = P_pred - K @ S @ K.T

    return x_upd, P_upd

# =============================================================================
#  5) Ana Simülasyon ve Çıktıların Çizilmesi
# =============================================================================

def main():
    # Zaman
    t = np.arange(0, sim_time, dt)

    # Basit bir giriş senaryosu:
    # ax sabit 0.5, ay=0, w=0.05 sabit
    ax_cmd = 0.5 * np.ones_like(t)
    ay_cmd = 0.0 * np.ones_like(t)
    w_cmd  = 0.05* np.ones_like(t)

    # "Gerçek" durumlar
    X_true = np.zeros((len(t), 5))
    x0_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [Vx, Vy, psi, Xg, Yg]
    X_true[0] = x0_true

    for k in range(len(t)-1):
        u_k = np.array([ax_cmd[k], ay_cmd[k], w_cmd[k]])
        X_true[k+1] = f(X_true[k], u_k, dt)

    # Ölçümler (GPS -> [Xg, Yg, Ve, Vn])
    Z_meas = np.zeros((len(t), 4))
    for k in range(len(t)):
        z_clean = h(X_true[k])
        noise = np.random.multivariate_normal(np.zeros(4), R)
        Z_meas[k] = z_clean + noise

    # EKF ve UKF başlatma
    xEKF = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    PEKF = np.eye(5) * 10.0

    xUKF = xEKF.copy()
    PUKF = np.eye(5) * 10.0

    X_est_EKF = np.zeros((len(t), 5))
    X_est_UKF = np.zeros((len(t), 5))
    X_est_EKF[0] = xEKF
    X_est_UKF[0] = xUKF

    # Filtre döngüsü
    for k in range(len(t)-1):
        u_k = np.array([ax_cmd[k], ay_cmd[k], w_cmd[k]])

        # --- EKF ---
        x_pred_ekf, P_pred_ekf = ekf_predict(xEKF, PEKF, u_k)
        x_upd_ekf, P_upd_ekf   = ekf_update(x_pred_ekf, P_pred_ekf, Z_meas[k+1])
        xEKF, PEKF = x_upd_ekf, P_upd_ekf
        X_est_EKF[k+1] = xEKF

        # --- UKF ---
        x_pred_ukf, P_pred_ukf, sigmas_pred, Wm, Wc = ukf_predict(xUKF, PUKF, u_k)
        x_upd_ukf, P_upd_ukf = ukf_update(x_pred_ukf, P_pred_ukf,
                                          sigmas_pred, Z_meas[k+1], Wm, Wc)
        xUKF, PUKF = x_upd_ukf, P_upd_ukf
        X_est_UKF[k+1] = xUKF

    # =============================================================================
    #  A) Performans için Hata Hesabı (ilk örnekteki gibi)
    # =============================================================================
    # Konum hatası
    ekf_pos_error = np.sqrt((X_est_EKF[:,3] - X_true[:,3])**2 
                            + (X_est_EKF[:,4] - X_true[:,4])**2)
    ukf_pos_error = np.sqrt((X_est_UKF[:,3] - X_true[:,3])**2
                            + (X_est_UKF[:,4] - X_true[:,4])**2)

    # Vx hatası
    ekf_vx_error = X_est_EKF[:,0] - X_true[:,0]
    ukf_vx_error = X_est_UKF[:,0] - X_true[:,0]

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(t, ekf_pos_error, label='EKF Position Error')
    plt.plot(t, ukf_pos_error, label='UKF Position Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Position Error [m]')
    plt.suptitle('Position and Horizontal Velocity Error Comparison of Filters vs True States')
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(t, ekf_vx_error, label='EKF Vx Error')
    plt.plot(t, ukf_vx_error, label='UKF Vx Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Vx Error [m/s]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print("EKF ortalama konum hatası =", np.mean(ekf_pos_error))
    print("UKF ortalama konum hatası =", np.mean(ukf_pos_error))
    print("EKF ortalama Vx hatası    =", np.mean(np.abs(ekf_vx_error)))
    print("UKF ortalama Vx hatası    =", np.mean(np.abs(ukf_vx_error)))

    # =============================================================================
    #  B) "Fig. 2" Tarzı: Vx, Vy, Psi çizimi
    # =============================================================================
    plt.figure(figsize=(9,7))

    # Vx
    plt.subplot(3,1,1)
    plt.plot(t, X_est_UKF[:,0], label='UKF')
    plt.plot(t, X_est_EKF[:,0], label='EKF')
    plt.plot(t, X_true[:,0],    label='Gerçek')
    plt.ylabel(r'$V_x$ [m/s]')
    plt.legend()
    plt.grid(True)

    # Vy
    plt.subplot(3,1,2)
    plt.plot(t, X_est_UKF[:,1], label='UKF')
    plt.plot(t, X_est_EKF[:,1], label='EKF')
    plt.plot(t, X_true[:,1],    label='Gerçek')
    plt.ylabel(r'$V_y$ [m/s]')
    plt.legend()
    plt.grid(True)

    # Psi
    plt.subplot(3,1,3)
    plt.plot(t, X_est_UKF[:,2], label='UKF')
    plt.plot(t, X_est_EKF[:,2], label='EKF')
    plt.plot(t, X_true[:,2],    label='Gerçek')
    plt.ylabel(r'$\psi$ [rad]')
    plt.xlabel('time [s]')
    plt.legend()
    plt.grid(True)
    plt.suptitle('Comparison of True States, EKF and UKF Performance')
    plt.tight_layout()
    plt.show()

    # =============================================================================
    #  C) "Fig. 3" Tarzı: X_g konumu çizimi
    # =============================================================================
    plt.figure()
    plt.plot(t, X_est_UKF[:,3], label='X_UKF')
    plt.plot(t, Z_meas[:,0],    label='X_GPS (measure)', linestyle='--')
    plt.plot(t, X_est_EKF[:,3], label='X_EKF')
    plt.ylabel(r'$x_g$ [m]')
    plt.xlabel('time [s]')
    plt.legend()
    plt.title('Estimation of Vehicle Positioning with EKF and UKF')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
