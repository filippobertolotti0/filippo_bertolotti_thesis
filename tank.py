import energym
weather = "CH_BS_Basel"
env = energym.make("SwissHouseRSlaTank-v0", weather=weather, simulation_days=20)
inputs = env.get_inputs_names()
outputs = env.get_outputs_names()
print("inputs:", inputs)
print("outputs:", outputs)