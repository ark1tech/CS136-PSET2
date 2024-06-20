import numpy as np
import math
import plotly.graph_objects as go

def runge_kutta_4(f, x0, y0, z0, t0, t_end, dt):
    n = math.ceil((t_end - t0) / dt)
    
    t = np.zeros(n+1)
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    z = np.zeros(n+1)
    
    x[0] = x0
    y[0] = y0
    z[0] = z0
    t[0] = t0
    
    for i in range(n):  
        k1T, k1B, k1M = f(x[i], y[i], z[i])
        k2T, k2B, k2M = f(x[i] + 0.5*dt*k1T, y[i] + 0.5*dt*k1B, z[i] + 0.5*dt*k1M)
        k3T, k3B, k3M = f(x[i] + 0.5*dt*k2T, y[i] + 0.5*dt*k2B, z[i] + 0.5*dt*k2M)
        k4T, k4B, k4M = f(x[i] + dt*k3T, y[i] + dt*k3B, z[i] + dt*k3M)
        
        x[i+1] = x[i] + (dt/6)*(k1T + 2*k2T + 2*k3T + k4T)
        y[i+1] = y[i] + (dt/6)*(k1B + 2*k2B + 2*k3B + k4B)
        z[i+1] = z[i] + (dt/6)*(k1M + 2*k2M + 2*k3M + k4M)
        t[i+1] = t[i] + dt
        
    return t, x, y, z

def system(T, B, M):
    weight = 70
    bodyfat = 0.20

    dTdt = -10*(1/weight)*T + 0.05*bodyfat*B
    dBdt = 3.8*(1/weight)*T - 0.11*bodyfat*B
    dMdt = ((1/weight) - bodyfat*(1/weight)) * 3.8 * T - (bodyfat-(bodyfat ** 2))* 0.11 * B

    return dTdt, dBdt, dMdt

# Initial conditions
dosage = 10.0
t0 = 5.0
t_end = 72.0
dt = 0.01

# Convert dosage (mL) to serum concentration (mcg/L) for T0
T0 = dosage

# Solve the system using the Runge-Kutta method
t, x, y, z = runge_kutta_4(system, T0, 0, 0, t0, t_end, dt)

# Plot the results
fig = go.Figure()
fig.add_traces([
    go.Scatter(x=t, y=x, mode='lines', marker = {'color' : 'red'}, name="Trapezius"),
    go.Scatter(x=t, y=y, mode='lines', marker = {'color' : 'blue'}, name="Bloodstream"),
    go.Scatter(x=t, y=z, mode='lines', marker = {'color' : 'green'}, name="Other muscles")
])
fig.update_layout(
    title_text='Clenbuterol presence in body over time',
    xaxis_title='Time (hours)',
    yaxis_title='Concentration (mcg/L)',
    height=1080*0.5,
    width=1920*0.6,
    font_family="CMU Serif",
    font_size=15,
    title_font_size=25,
    font_color="#0e0f11",
    margin=dict(t=120, b=80)
)
fig.show()