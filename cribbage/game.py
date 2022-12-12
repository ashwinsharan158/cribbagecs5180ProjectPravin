from random import choice
import torch
import torch.nn.functional as F
from .replaybuffer import replaybuffer
from .card import Deck
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque


class Hand:
    def __init__(self, dealer, pone):
        """
        Create a new hand

        Parameters
        ----------
        dealer: cribbage.Player
            The player who is the dealer 
        
        pone: cribbage.Player
            The player who is the opponent 
        """

        self.dealer = dealer
        self.pone = pone
        self.cardOnTable = []  # All card
        self.previousPlays = []  # Current Game
        self.turn_card = None
        self.turn_index = 0
        self.turn_map = {0: self.pone, 1: self.dealer}
        self.plays = []
        self.count = 0
        self.go_has_been_said = False

    def getObs(self):
        total_cards = self.cardOnTable + self.previousPlays
        x = list(map(lambda x: x.run_val, total_cards))
        for _ in range(len(x), 8):
            x.append(0)
        card_in_hand = self.pone.hand_run_val()
        cardInHand = []
        for y in card_in_hand:
            cardInHand.append(y)
        for _ in range(len(cardInHand), 4):
            cardInHand.append(0)
        state = x + cardInHand
        # state.append(self.count)
        state.append(1)
        # state = F.one_hot(torch.tensor(state), num_classes=14)
        return state

    def getObsDealer(self):
        total_cards = self.cardOnTable + self.previousPlays
        x = list(map(lambda x: x.run_val, total_cards))
        for _ in range(len(x), 8):
            x.append(0)
        card_in_hand = self.dealer.hand_run_val()
        cardInHand = []
        for y in card_in_hand:
            cardInHand.append(y)
        for _ in range(len(cardInHand), 4):
            cardInHand.append(0)
        state = x + cardInHand
        # state.append(self.count)
        state.append(1)
        # state = F.one_hot(torch.tensor(state), num_classes=14)
        return state

    def run(self, data):
        """Run the entire hand"""

        self.deal()
        # self.discards()
        print(f'Turn card {self.turn_card}')
        self.counting(data)
        self.count_hands()
        self.clean_up()

    def deal(self):
        """Create a new deck and deal cards to players"""

        deck = Deck()
        # self.dealer.hand = list(deck.draw(6))
        # self.pone.hand = list(deck.draw(6))
        # self.turn_card = next(deck.draw(1))

        self.dealer.hand = list(deck.draw(4))
        self.pone.hand = list(deck.draw(4))
        self.turn_card = next(deck.draw(1))

    def discards(self):
        """Get discards from both players and add them to crib"""

        d1 = self.dealer.discards()
        d2 = self.pone.discards()
        self.dealer.crib = d1 + d2

    def reset(self):
        # self.discards()
        # print(f'Turn card {self.turn_card}')
        # self.counting(data)
        # self.count_hands()
        self.clean_up()
        self.deal()
        return

    def step(self, action):
        """Starting with two players with at least one card between them,
        and a count of 0, start the counting portion of the game given
        information about who"""
        done = False
        # if not self.dealer.hand:
        #    print('dealer has no cards')
        # if not self.pone.hand:
        #    print('pone has no cards')

        # can be `"Go!"` or a card object
        if isinstance(action, str):
            if self.go_has_been_said:
                # print('"Go!" has already been said, so starting a new count at 0')
                done = True
                self.count = 0
                self.cardOnTable = self.cardOnTable + self.previousPlays
                self.previousPlays = []
                self.go_has_been_said = False
            else:
                # print('Go by player!')
                self.go_has_been_said = True
        else:
            # print("Player Action :", action)
            self.previousPlays.append(action)
            self.count += action.value
            # score the play
            # needs a rework of `score_hand` to accept < 4 cards and no turn card
            if self.count == 31:
                # print('counted to 31, point for Player')
                self.pone.peg(1)
                done = True
                self.count = 0
                self.cardOnTable = self.cardOnTable + self.previousPlays
                self.previousPlays = []

        # print('Cards on table and their count:', self.previousPlays, self.count)
        action = self.dealer.play(self.count, self.previousPlays)
        if isinstance(action, str):
            if self.go_has_been_said:
                # print('"Go!" has already been said, so starting a new count at 0')
                done = True
                self.count = 0
                self.cardOnTable = self.cardOnTable + self.previousPlays
                self.previousPlays = []
                self.go_has_been_said = False
            else:
                # print('Go by dealer!')
                self.go_has_been_said = True
        else:
            # print('Dealer Action :', action)
            self.previousPlays.append(action)
            self.count += action.value
            # score the play
            # needs a rework of `score_hand` to accept < 4 cards and no turn card
            if self.count == 31:
                # print('counted to 31, point for Dealer')
                self.dealer.peg(1)
                done = True
                self.count = 0
                self.cardOnTable = self.cardOnTable + self.previousPlays
                self.previousPlays = []
        return done

    def stepSelfPlay(self, obs, action):
        """Starting with two players with at least one card between them,
        and a count of 0, start the counting portion of the game given
        information about who"""
        done = False

        # can be `"Go!"` or a card object
        if isinstance(action, str):
            if self.go_has_been_said:
                # print('"Go!" has already been said, so starting a new count at 0')
                done = True
                self.count = 0
                self.cardOnTable = self.cardOnTable + self.previousPlays
                self.previousPlays = []
                self.go_has_been_said = False
            else:
                # print('Go by player!')
                self.go_has_been_said = True
        else:
            # print("Player Action :", action)
            self.previousPlays.append(action)
            self.count += action.value
            # score the play
            # needs a rework of `score_hand` to accept < 4 cards and no turn card
            if self.count == 31:
                # print('counted to 31, point for Player')
                self.pone.peg(1)
                done = True
                self.count = 0
                self.cardOnTable = self.cardOnTable + self.previousPlays
                self.previousPlays = []

        # print('Cards on table and their count:', self.previousPlays, self.count)
        state = self.getObsDealer()
        action, best_action = self.dealer.play(self.count, self.previousPlays, state)
        # action = self.dealer.play(self.count, self.previousPlays, obs)
        if isinstance(action, str):
            if self.go_has_been_said:
                # print('"Go!" has already been said, so starting a new count at 0')
                done = True
                self.count = 0
                self.cardOnTable = self.cardOnTable + self.previousPlays
                self.previousPlays = []
                self.go_has_been_said = False
            else:
                # print('Go by dealer!')
                self.go_has_been_said = True
        else:
            # print('Dealer Action :', action)
            self.previousPlays.append(action)
            self.count += action.value
            # score the play
            # needs a rework of `score_hand` to accept < 4 cards and no turn card
            if self.count == 31:
                # print('counted to 31, point for Dealer')
                self.dealer.peg(1)
                done = True
                self.count = 0
                self.cardOnTable = self.cardOnTable + self.previousPlays
                self.previousPlays = []
        return done, best_action

    def count_to_31(self):
        """Starting with two players with at least one card between them, 
        and a count of 0, start the counting portion of the game given 
        information about who"""

        if not self.dealer.hand:
            print('dealer has no cards')
        if not self.pone.hand:
            print('pone has no cards')

        count = 0
        turn = 0  # index of whose turn it is
        plays = []
        done = None
        go_has_been_said = False
        l = []
        while not done:
            print('counting:', plays, count)

            # player whose turn it is plays 
            player = self.turn_map[turn]
            hand = player.hand.copy()
            my_play = player.play(count, plays)  # can be `"Go!"` or a card object
            # ct = count
            plays_val = []
            hand_val = []
            for x in plays:
                plays_val.append(x.value)
            for x in hand:
                hand_val.append(x.value)
            my_play_val = my_play
            if not isinstance(my_play, str):
                my_play_val = my_play.value
            l.append([player.name, plays_val, count, hand_val, my_play_val])
            _ = self.pone.count_hand(self.turn_card)
            print('####### pone', self.pone, self.pone.table, _)

            _ = self.dealer.count_hand(self.turn_card)
            print('######### dealer', self.dealer, self.dealer.table, _)

            if my_play not in hand:
                print(hand)

            if isinstance(my_play, str):
                if go_has_been_said:
                    print('"Go!" has already been said, so starting a new count at 0')
                    done = True
                    count = 0
                    break

                print('that"s a "go", switching turns')
                go_has_been_said = True
            else:
                print(player, 'played', my_play)
                count += my_play.value
                plays.append(my_play)
                # score the play
                # needs a rework of `score_hand` to accept < 4 cards and no turn card 
                if count == 31:
                    print('counted to 31, point for', player)
                    player.peg(1)
                    done = True
                    count = 0

            turn = turn ^ 1
        return l

    def counting(self, data):
        print(f'Counting starts with {self.pone}')
        while len(self.dealer.hand) + len(self.pone.hand) > 0:
            data.append(self.count_to_31())

    def count_hands(self):
        print('Counting hands')

        _ = self.pone.count_hand(self.turn_card)
        print('pone', self.pone, self.pone.table, _)

        _ = self.dealer.count_hand(self.turn_card)
        print('dealer', self.dealer, self.dealer.table, _)

        # _ = self.dealer.count_crib(self.turn_card)
        # print('crib', self.dealer, self.dealer.crib, _)

    def clean_up(self):
        self.dealer.table = []
        self.pone.table = []
        self.cardOnTable = []
        self.previousPlays = []
        self.count = 0
        self.go_has_been_said = False


class Game:
    def __init__(self, A, B, deal=None):
        """Create a new Game object from two Player instances
        
        Parameters
        ----------
        A: cribbage.players.Player
            A cribbage player 
        B: cribbage.players.Player
            A cribbage player
        
        Raises
        ------
        WinGame
            When game has been won by a player 
        """

        self.data = []
        self.A = A
        self.B = B
        if deal is None:
            self.deal = choice((0, 1))
            print(
                f"############\n# Cribbage # \n############ \nStarting a new game with dealer \"{[self.A, self.B][self.deal]}\" and opponent \"{[self.A, self.B][self.deal ^ 1]}\"")

    def runfinal(self, data):
        # Create hand
        hand = Hand(self.A, self.B)
        # Create neural network instance
        hand.pone.updateAgent()
        agent = hand.pone.agent
        replay_buffer = replaybuffer()
        pone_name = hand.pone.name
        ep = 1
        r = []
        loss_values = []
        sum = 0
        AIscore = 0
        randomAgentscore = 0
        pone = 0
        dealer = 0

        i = 0
        trials =[]
        for _ in range(1):
            while ep <= 6000:
                i = i + 1
                hand.reset()
                while True:
                    curr_run_val = hand.pone.hand_run_val()
                    state = hand.getObs()
                    action = hand.pone.play(hand.count, hand.previousPlays, state)
                    if not isinstance(action, str) and action.run_val not in curr_run_val:
                        l = (state, action.run_val, state, 0, 0)
                        data.append(l)
                        continue
                    done = hand.step(action)
                    # print(hand.cardOnTable, state, cardInHand, action, hand.cardOnTable, hand.previousPlays, 0)
                    reward = 0
                    actVal = 14
                    if not isinstance(action, str):
                        actVal = action.run_val
                    next_obs = hand.getObs()
                    l = (state, actVal, next_obs, reward, 0)
                    if len(hand.dealer.hand) + len(hand.pone.hand) == 0:
                        x = hand.pone.count_hand(hand.turn_card)
                        AIscore = AIscore + x
                        y = hand.dealer.count_hand(hand.turn_card)
                        randomAgentscore = randomAgentscore + y
                        # AIscore = hand.pone.count_hand(hand.turn_card)
                        # randomAgentscore = hand.dealer.count_hand(hand.turn_card)
                        print("Scores : ", AIscore, randomAgentscore)
                        reward = x - y
                        l = (state, actVal, next_obs, reward, 1)
                        # data.append(l)
                        replay_buffer.addEpisode(l)
                        break
                    # data.append(l)
                    replay_buffer.addEpisode(l)
                # print(data)
                # replay_buffer.addEpisode(data)
                # data = []
                # print(replay_buffer)

                if AIscore >= 121 or randomAgentscore >= 121:
                    ep = ep + 1
                    if AIscore > randomAgentscore:
                        sum = sum + AIscore - randomAgentscore
                        r.append(sum)
                        pone = pone + 1
                    else:
                        sum = sum + AIscore - randomAgentscore
                        r.append(sum)
                        dealer = dealer + 1

                    AIscore = 0
                    randomAgentscore = 0
                    if (ep % 1000) == 0:
                        plt.figure(1)
                        plt.plot(r, 'b')
                        plt.ylabel('Cumulative Net reward')
                        plt.xlabel('Episodes')
                        plt.savefig("./output/" + pone_name + "/_" + str(ep) + "_rewards.png")
                        plt.grid(True)
                    if (ep % 5000) == 0:
                        plt.show()
                    if (ep % 800) == 0:
                        plt.figure(2)
                        plt.plot(loss_values, 'r')
                        plt.ylabel('Network Loss')
                        plt.xlabel('Episodes')
                        plt.savefig("./output/" + pone_name + "/_" + str(ep) + "_loss.png")
                        # plt.show()
                        plt.figure(3)
                        plt.plot(loss_values[5:], 'r')
                        plt.ylabel('Network Loss')
                        plt.xlabel('Episodes')
                        plt.savefig("./output/" + pone_name + "/_" + str(ep) + "_No1loss.png")
                    #  plt.show()
                    if ep > 500 and (ep % 4) == 0:
                        loss = agent.update_behavior_policy(replay_buffer.sampleBatch(64))
                        loss_values.append(loss)
                    if ep > 500 and (ep % 1000) == 0:
                        agent.update_target_policy()
            trials.append(r)
            # print("Total rewards:", r)
            # print(replay_buffer.currSize)

        l=[]
        for x in range(6000):
            s=0
            for y in range(5):
                s = s+ trials[y][x]
            l.append(s/5)
        plt.figure(6)
        plt.plot(r, 'b')
        plt.ylabel('Average Cumulative Net reward')
        plt.xlabel('Episodes')
        plt.savefig("./output/" + pone_name + "/_" + str(ep) + "_rewards1.png")
        plt.grid(True)
        plt.show()
        agent.save_model()
        torch.save(torch.tensor(r), "./output/" + pone_name + "/_" + str(ep) + '_rewards.pt')
        torch.save(torch.tensor(loss), "./output/" + pone_name + "/_" + str(ep) + '_loss.pt')
        print("Final wins:", pone, dealer)

        return r

    def runSelfPlay(self, selfPlay, data):

        # Create hand
        hand = Hand(self.A, self.B)
        # Create neural network instance
        hand.pone.updateAgent()
        hand.dealer.updateAgent()
        agent1 = hand.pone.agent
        agent2 = hand.dealer.agent
        # agent1.behavior_network.load_state_dict(torch.load(f'{"./DRQN.pt"}'))
        # agent1.target_network.load_state_dict(torch.load(f'{"./DRQN.pt"}'))
        replay_buffer = replaybuffer()
        pone_name = hand.pone.name
        ep = 1
        r = []
        loss_values = []
        sum = 0
        AIscore = 0
        randomAgentscore = 0
        reservoir_Buffer = replaybuffer()
        best_response = False
        i = 0
        while ep < 5000:
            i = i + 1
            hand.reset()
            while True:
                curr_run_val = hand.pone.hand_run_val()
                state = hand.getObs()
                action, best_response1 = hand.pone.play(hand.count, hand.previousPlays, state)

                if not isinstance(action, str) and action.run_val not in curr_run_val:
                    l = (state, action.run_val, state, 0, 0)
                    data.append(l)
                    continue

                obs = hand.getObs()
                done, best_response2 = hand.stepSelfPlay(obs, action)

                # print(hand.cardOnTable, state, cardInHand, action, hand.cardOnTable, hand.previousPlays, 0)
                reward = 0
                actVal = 14
                if not isinstance(action, str):
                    actVal = action.run_val

                next_obs = hand.getObs()
                l = (state, actVal, next_obs, reward, 0)
                if len(hand.dealer.hand) + len(hand.pone.hand) == 0:
                    x = hand.pone.count_hand(hand.turn_card)
                    AIscore = AIscore + x
                    y = hand.dealer.count_hand(hand.turn_card)
                    randomAgentscore = randomAgentscore + y
                    print("Scores : ", AIscore, randomAgentscore)
                    reward = x - y
                    l = (state, actVal, next_obs, reward, 1)
                    replay_buffer.addEpisode(l)
                    if best_response:
                        reservoir_Buffer.addEpisode(l)
                    break
                replay_buffer.addEpisode(l)
                # if best_response:
                reservoir_Buffer.addEpisode(l)

            if AIscore >= 121 or randomAgentscore >= 121:
                ep = ep + 1
                if AIscore > randomAgentscore:
                    sum = sum + AIscore - randomAgentscore
                    r.append(sum)
                else:
                    sum = sum + AIscore - randomAgentscore
                    r.append(sum)
                # Episode ends reset score
                AIscore = 0
                randomAgentscore = 0

                if (ep % 1000) == 0:
                    plt.figure(1)
                    plt.plot(r, 'b')
                    plt.ylabel('Net reward')
                    plt.xlabel('Episodes')
                    plt.savefig("./output/" + pone_name + "/_" + str(ep) + "_rewards.png")
                    plt.grid(True)
                # plt.show()
                if (ep % 800) == 0:
                    plt.figure(2)
                    plt.plot(loss_values, 'r')
                    plt.ylabel('Network Loss')
                    plt.xlabel('Episodes')
                    plt.savefig("./output/" + pone_name + "/_" + str(ep) + "_loss.png")
                    #  plt.show()
                    plt.figure(3)
                    plt.plot(loss_values[5:], 'r')
                    plt.ylabel('Network Loss')
                    plt.xlabel('Episodes')
                    plt.savefig("./output/" + pone_name + "/_" + str(ep) + "_No1loss.png")
                #  plt.show()
                if ep > 500 and (ep % 4) == 0:
                    loss = agent1.update_behavior_policy(replay_buffer.sampleBatch(128))
                    loss_values.append(loss)
                    loss = agent2.update_behavior_policy(replay_buffer.sampleBatch(128))
                    if best_response1:
                        agent1.update_policy(reservoir_Buffer.sampleBatch(128))
                    if best_response2:
                        agent2.update_policy(reservoir_Buffer.sampleBatch(128))

                if ep > 500 and (ep % 2000) == 0:
                    agent1.update_target_policy()
                    agent2.update_target_policy()
        agent1.save_model(1)
        agent2.save_model(2)
        torch.save(torch.tensor(r), "./output/" + pone_name + "/_" + str(ep) + '_rewards.pt')
        torch.save(torch.tensor(loss), "./output/" + pone_name + "/_" + str(ep) + '_loss.pt')
        return r

    def run(self, data):
        while True:
            if self.deal == 0:
                hand = Hand(self.A, self.B)
                hand.run(data)
                # self.deal = 1
            else:
                hand = Hand(self.B, self.A)
                hand.run(data)
                self.deal = 0
