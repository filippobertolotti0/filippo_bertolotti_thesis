import energym
from scipy import signal
import matplotlib.pyplot as plt
    
if __name__ == "__main__":
    weather = "CH_BS_Basel"
    env = energym.make("SwissHouseRSlaW2W-v0", weather=weather, simulation_days=20)
    
    steps = 288*5
    out_list = []
    outputs = env.get_output()
    controls = []
    hour = 0
    for i in range(steps):
        control = {}
        control['u'] = [0.5*(signal.square(0.1*i)+1.0)]
        controls +=[ {p:control[p][0] for p in control} ]
        outputs = env.step(control)
        _,hour,_,_ = env.get_date()
        out_list.append(outputs)
        
    import pandas as pd
    out_df = pd.DataFrame(out_list)
    
    f, (ax1,ax2,ax3) = plt.subplots(3,figsize=(10,15))#

    ax1.plot(out_df['temRoo.T']-273.15, 'r')
    ax1.plot(out_df['sla.heatPortEmb[1].T']-273.15, 'b--')
    ax1.plot(out_df['heaPum.TEvaAct']-273.15, 'orange')
    ax1.set_ylabel('Temp')
    ax1.set_xlabel('Steps')

    ax2.plot(out_df['TOut.T']-273.15, 'r')
    ax2.set_ylabel('Temp')
    ax2.set_xlabel('Steps')

    ax3.plot(out_df['heaPum.QCon_flow'], 'g')
    ax3.set_ylabel('Energy')
    ax3.set_xlabel('Steps')

    plt.subplots_adjust(hspace=0.4)

    plt.show()