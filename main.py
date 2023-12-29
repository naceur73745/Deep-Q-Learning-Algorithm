"""#Agent"""
import torch 
from DQAgent import Agent  
from env import Prisoners
import numpy as np 

def plot_bar_chart_function(rate_of_cooperation, rate_of_defection, title_under_first_plot, title_under_second_plot, title, y_label):
    x = [1, 2]  # x-axis values

    # Heights of the bars
    heights = [rate_of_cooperation, rate_of_defection]

    # Labels for the bars
    labels = [title_under_first_plot, title_under_second_plot]

    # Plotting the bar chart
    plt.bar(x, heights)

    # Adding labels and title
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x, labels)

    # Displaying the plot
    plt.show()


#this  calculates the rewards  of player the reward list should be for only one round

def calculate_reward(reward_list , player_index ):

  reward  = 0
  for state in  reward_list :
    reward+= state[player_index ]
  return reward


#this function used to define the cooperation rate and  and defction rate for player in round and it works only on oponent who alw coop and defct

def coop_defect_rate (state_list ,strategy  ) :

  #we cann add the others startegy also  to test
  right_decision =  0
  false_decision = 0
  for state  in state_list   :

   if strategy == 1: #stands for opent is always defecting  the right decison  will be (1,1)
      if state ==(1,1) :
        right_decision = right_decision+1
      else :
        false_decision = false_decision+1
   else :

      if state ==(1,0) : #in case the oponent always coop  the right decison will eb to alwys defect

        right_decision = right_decision+1
      else :
        false_decision = false_decision+1

  return right_decision , false_decision, strategy

#create an  agent
  def Create_agent(input_dim ,dim1 , dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda):
    return Agent(input_dim ,dim1, dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda)
#this plott rewqrds over epochen

#plott reword over epochen
def plot(reward_over_epochen , title  , label , x_label  , y_label ):

    epochs = range(1, len(reward_over_epochen) + 1)
    plt.plot(epochs, reward_over_epochen, 'b', label= label  )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


#Function 's


    epochs = range(1, len(reward_over_epochen) + 1)
    plt.plot(epochs, reward_over_epochen, 'b', label= label  )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()



def convert_function(state) :
  if state  == (0,0) :
    return  0
  elif  state  == (1,0) :
    return  1
  elif state == (0,1) :
    return 2
  else  :
    return  3
def test_function (epsiode_len , n_games ,values, agent):

  env = Prisoners(epsiode_len ,n_games)
  states= []
  rewards=[]

  for Round in  range (n_games ):

    state = env.reset()


    for i in range  (epsiode_len ) :
      print(f"the current state is {state}")

      state  = agent.convert_function(state )

      action = np.argmax(agent.Q_table[state,:])

      new_state  , reward , done , info =  env.evaluate(action ,values  )


      states.append(new_state)

      rewards.append(info )

      state = new_state

  return states  , rewards

"""#Train Agent"""

import matplotlib.pyplot as plt

def Create_agent(input_dim ,dim1 , dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  ):
  return Agent(input_dim ,dim1, dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  )

def train_function(episode_len  , n_games ,input_dim ,dim1,dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec   ):
  env = Prisoners(episode_len,n_games)
  agent  = Create_agent(input_dim ,dim1,dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec    )
  for Round in range  (n_games ):

    state = env.reset()
    step = 0
    while env.done == False   :


      action    = agent.choose_action(state)

      new_state  , reward, done , info =  env.step(action)


      agent.mem.store_action(state , new_state,action ,reward[0] ,done )

      agent.learn()

      state = new_state

      print( f" Round  : {Round} , step:{step} ")
      step+= 1



  return env.state_total , env.reward_total , env.chooosed_startegy_each_round ,agent


 #need more exploration
 #def Create_agent(input_dim ,dim1 , dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  ):

states  , rewards  , startegies,agent   = train_function(100,500 , 2,128,512,2,0.0001,128,2048,0.99,0.9)
print( "train mode ")
print(states )
print(rewards )
print(startegies )
print("\n")

#function for evaluation
#this function plotts the cooperation rate and defection rate for a player
def plot_bar_chart(rate_of_cooperation, rate_of_defection, title_under_first_plot, title_under_second_plot, title, y_label):
    x = [1, 2]  # x-axis values

    # Heights of the bars
    heights = [rate_of_cooperation, rate_of_defection]

    # Labels for the bars
    labels = [title_under_first_plot, title_under_second_plot]

    # Plotting the bar chart
    plt.bar(x, heights)

    # Adding labels and title
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x, labels)

    # Displaying the plot
    plt.show()


#this  calculates the rewards  of player the reward list should be for only one round

def calculate_reward(reward_list , player_index ):

  reward  = 0
  for state in  reward_list :
    reward+= state[player_index ]
  return reward


#this function used to define the cooperation rate and  and defction rate for player in round and it works only on oponent who alw coop and defct

def coop_defect_rate (state_list ,strategy  ) :

  #we cann add the others startegy also  to test
  right_decision =  0
  false_decision = 0
  for state  in state_list   :

   if strategy == 1: #stands for opent is always defecting  the right decison  will be (1,1)
      if state ==(1,1) :
        right_decision = right_decision+1
      else :
        false_decision = false_decision+1
   else :

      if state ==(1,0) : #in case the oponent always coop  the right decison will eb to alwys defect

        right_decision = right_decision+1
      else :
        false_decision = false_decision+1

  return right_decision , false_decision, strategy

#create an  agent
  def Create_agent(input_dim ,dim1 , dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda):
    return Agent(input_dim ,dim1, dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda)
#this plott rewqrds over epochen

#plott reword over epochen
def plot(reward_over_epochen , title  , label , x_label  , y_label ):

    epochs = range(1, len(reward_over_epochen) + 1)
    plt.plot(epochs, reward_over_epochen, 'b', label= label  )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def plot_Reward(reward_over_epochen1,reward_over_epochen2  , title  , label1, label2 , x_label  , y_label ):


    epochs = range(1, len(reward_over_epochen1) + 1)
    plt.plot(epochs, reward_over_epochen1, 'b', label= label1 )
    plt.plot(epochs, reward_over_epochen2, 'r', label= label2  )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()




def test_function (epsiode_len , n_games ,values, agent):

  env = Prisoners(epsiode_len ,n_games)
  states= []
  rewards=[]

  for Round in  range (n_games ):

    state = env.reset()

    for i in range  (epsiode_len ) :

      state  = agent.convert_function(state )

      action = np.argmax(agent.Q_table[state,:])

      new_state  , reward , done , info =  env.evaluate(action ,values  )

      states.append(new_state)

      rewards.append(info )

      state = new_state

  return states  , rewards



def count_coop_defect_rate (list_of_state):
      player1_defection_rate = 0
      palyer1_cooperation_rate= 0
      #player 2
      player2_defection_rate = 0
      palyer2_cooperation_rate= 0

      for state in list_of_state :
        if state[0] == 1  :
          player1_defection_rate+= 1
        else  :
          palyer1_cooperation_rate+= 1

        if state[1] == 1  :
          player2_defection_rate+= 1
        else  :
          palyer2_cooperation_rate+= 1
      return  player1_defection_rate , palyer1_cooperation_rate  , player2_defection_rate , palyer2_cooperation_rate

  #"define a function for comaparison "



import matplotlib.pyplot as plt

def generate_bar_chart(player1_rates, player2_rates):
    import matplotlib.pyplot as plt

    # Extract cooperation and defection rates for Player 1
    player1_cooperation_rate = player1_rates[0]
    player1_defection_rate = player1_rates[1]

    # Extract cooperation and defection rates for Player 2
    player2_cooperation_rate = player2_rates[0]
    player2_defection_rate = player2_rates[1]

    # Bar positions for Player 1 and Player 2
    player1_positions = [0, 1]
    player2_positions = [3, 4]

    # Heights of the bars
    player1_heights = [player1_cooperation_rate, player1_defection_rate]
    player2_heights = [player2_cooperation_rate, player2_defection_rate]

    # Bar labels
    player1_labels = ['Cooperation', 'Defection']
    player2_labels = ['Cooperation', 'Defection']

    # Plotting the bar chart
    plt.bar(player1_positions, player1_heights, align='center', alpha=0.5, label='Player 1')
    plt.bar(player2_positions, player2_heights, align='center', alpha=0.5, label='Player 2')

    # Adjusting the x-tick positions and labels
    plt.xticks([0, 1, 3, 4], player1_labels + player2_labels)

    plt.ylabel('Rate')
    plt.title('Cooperation and Defection Rates')

    # Adding legend
    plt.legend()

    # Display the chart
    plt.show()



"""the agent is  trainiert  unter  Deep Q  learning
as we can notice the agent learned to always coop  with agent who  always coop  !!

"""

import matplotlib.pyplot as plt
import random

def plot(total_wins ,total_sum):

    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(total_wins)
    plt.plot(total_sum)
    plt.ylim(ymin=0)
    plt.text(len(total_wins)-1, total_wins[-1], str(total_wins[-1]))
    plt.text(len(total_sum)-1, total_sum[-1], str(total_sum[-1]))
    plt.show(block=False)
    plt.pause(.1)


def test_function (epsiode_len , n_games ,values, agent):

  env = Prisoners(epsiode_len ,n_games)
  states= []
  rewards=[]
  player_reward = []
  player_reward2= []
  win_round = 0
  winner_pro_round = []
  winner_pro_state = []



  for Round in  range (n_games ):

    state = env.reset() #state randomly
    sum = 0
    sum2= 0
    current_player1_score  = 0
    current_player2_score  = 0
    winnner_state = 0

    for i in range  (epsiode_len ) :

      action    = agent.network.forward(torch.tensor(state ,dtype=torch.float32))
      action = action.tolist()
      action =np.argmax(action)

      new_state  , reward , done , info =  env.evaluate(action,values)

      sum+= reward
      sum2+=info[1]

      current_player1_score+= info[0]
      current_player2_score+= info[1]

      if info[0] >= info[1] :

          winnner_state += 1



      states.append(new_state)
      rewards.append(info )
      state = new_state
      player_reward.append(sum)
      player_reward2.append(sum2)

    if current_player1_score  > current_player2_score  :
      win_round+=1
    winner_pro_round.append(win_round)
    winner_pro_state.append(winnner_state)




  return  winner_pro_round  ,winner_pro_state
  #return states  , rewards  ,player_reward ,player_reward2 , winner_pro_round , winner_pro_state

winner_pro_round , winner_pro_state  ,  = test_function(1000 , 5 ,2 , agent)

plot(winner_pro_round ,winner_pro_state)

