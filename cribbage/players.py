from itertools import combinations
from random import choice, shuffle
import torch
import numpy as np
from tqdm import tqdm
from Network.DQNAgent import DQNAgent
from Network.DDQN import DDQNAgent
from Network.DRQNAgent import DRQNAgent
from .score import score_hand, score_count
from .card import Deck, Card
from Network.SelfPlayAgent import NFSPPolicyAgent


class WinGame(Exception):
    pass


class Player:
    """
    Here we define a base class for the various kinds of players in the 
    cribbage game. To define your own player, inherit from this class and 
    implement ``ask_for_input`` and ``ask_for_discards``
    """

    def __init__(self, name=None):
        self.name = name
        self.hand = []  # cards in player's hand
        self.table = []  # cards on table in front of player
        self.crib = []  # cards in player's crib
        self.score = 0
        self.agent = 0

    # Discards

    def ask_for_discards(self):
        """Should return two cards from the player"""
        raise Exception("You need to implement `ask_for_discards` yourself")

    def update_after_discards(self, discards):
        for discard in discards:
            self.hand.remove(discard)

    def discards(self):
        cards = self.ask_for_discards()
        self.update_after_discards(cards)
        return cards

    # Counting plays

    def ask_for_play(self, previous_plays):
        """Should return a single card from the player

        Private method"""
        raise Exception("You need to implement `ask_for_play` yourself")

    def update_after_play(self, play):
        """Private method"""

        self.table.append(play)
        self.hand.remove(play)

    def hand_run_val(self):
        y = sorted(self.hand, key=lambda c: c.run_val)
        return list(map(lambda x: x.run_val, y))

    def hand_val(self):
        y = sorted(self.hand, key=lambda c: c.run_val)
        return list(map(lambda x: x.value, y))

    def play(self, count, previous_plays):
        """Public method"""
        if not self.hand:
            print('>>> I have no cards', self)
            return "No cards!"
        elif all(count + card.value > 31 for card in self.hand):
            print(">>>", self, self.hand, "I have to say 'go' on that one")
            return "Go!"

        while True:
            card = self.ask_for_play(previous_plays)  # subclasses (that is, actual players) must implement this
            # print("Nominated card", card)
            if sum((pp.value for pp in previous_plays)) + card.value < 32:
                self.update_after_play(card)
                return card
            else:
                # `self.ask_for_play` has returned a card that would put the 
                # score above 31 but the player's hand contains at least one
                # card that could be legally played (you're not allowed to say
                # "go" here if you can legally play). How the code knows that 
                # the player has a legal move is beyond me
                print('>>> You nominated', card,
                      'but that is not a legal play given your hand. You must play if you can')
                return "Go!"

    # Scoring
    # score of player
    def peg(self, points):
        i = 1
        # self.score += points

    # if self.score > 120:
    #   self.win_game()

    def count_hand(self, turn_card):
        """Count the hand (which should be on the table)"""
        score = score_hand(self.table, turn_card)
        # score =  score_count(self.table)
        # self.peg(score)
        return score

    def count_crib(self, turn_card):
        """Count crib, with side effect of pegging the score"""
        score = score_hand(self.crib, turn_card)
        self.peg(score)
        return score

    def win_game(self):
        raise WinGame(f"""Game was won by {self}  """
                      f"""{self.score} {self.table}""")

    @property
    def sorted_hand(self):
        return sorted(self.hand, key=lambda c: c.index)

    def __repr__(self):
        if self.name:
            return self.name + f'(score={self.score})'
        return str(self.__class__) + f'(score={self.score})'


class RandomPlayer(Player):
    """
    A player who plays randomly
    """

    def ask_for_play(self, previous_plays):
        shuffle(self.hand)  # this handles a case when 0 is not a legal play
        return self.hand[0]

    def ask_for_discards(self):
        return self.hand[0:2]


class HumanPlayer(Player):
    """
    A human player. This class implements scene rendering and taking input from
    the command line 
    """

    def ask_for_play(self, previous_plays):
        """Ask a human for a card during counting"""

        d = dict(enumerate(self.hand, 1))
        print(f">>> Your hand ({self}):", " ".join([str(c) for c in self.hand]))

        while True:
            inp = input(">>> Card number to play: ") or "1"
            if len(inp) > 0 and int(inp) in d.keys():
                card = d[int(inp)]
                return card

    def ask_for_discards(self):
        """After deal, ask a human for two cards to discard to crib"""

        d = dict(enumerate(self.sorted_hand, 1))

        print('>>> Please nominate two cards to discard to the crib')
        print(f'>>> {d[1]} {d[2]} {d[3]} {d[4]} {d[5]} {d[6]}')
        discard_prompt = ">>> "

        while True:
            inp = input(discard_prompt) or "12"
            cards = [d[int(i)] for i in inp.replace(" ", "").replace(",", "")]
            if len(cards) == 2:
                print(f">>> Discarded {cards[0]} {cards[1]}")
                return cards


class GreedyAgentPlayer(Player):
    """
    "Expert systems" style AI player that systematically
    enumerates possible moves and chooses the move that
    maximizes its score after the move
    """

    def ask_for_discards(self):
        """
        For each possible discard, score and select
        highest scoring move. It will discard cards
        that create the least value.
        """

        print("cribbage: {} is choosing discards".format(self))
        deck = Deck().draw(52)
        potential_cards = [n for n in deck if n not in self.hand]
        bar = tqdm(total=226994)
        discards = []
        mean_scores = []
        for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
            inner_scores = []
            for pot in combinations(potential_cards, 3):  # 46 choose 3 == 15,180
                inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]))
                bar.update(1)
            inner_scores = np.array(inner_scores)
            discards.append(discard)
            mean_scores.append(inner_scores.mean())

        return list(discards[np.argmin(mean_scores)])

    def ask_for_play(self, previous_plays):
        """
        Calculate points for each possible play in your hand
        and choose the one that maximizes the points
        """

        scores = []
        plays = []
        for card in self.hand:
            plays.append(card)
            scores.append(score_count(previous_plays + [card]))
        max_index = np.argmax(scores)

        return plays[max_index]


class DRQNAgentPlayer(Player):
    """
    "Expert systems" style AI player that systematically
    enumerates possible moves and chooses the move that
    maximizes its score after the move
    """

    def updateAgent(self):
        parameters = {
            "action_dimension": 15 * 1,
            "observation_dimension": 13 * 14,
            "hidden_layer_dimension": 256,
            "learning_rate": 0.1,
            "epsilon": 0.1
        }
        self.agent = DRQNAgent(parameters)

    def play(self, count, previous_plays, obs):
        """Public method"""
        if not self.hand:
            print('>>> I have no cards', self)
            return "No cards!"
        elif all(count + card.value > 31 for card in self.hand):
            print(">>>", self, self.hand, "I have to say 'go' on that one")
            return "Go!"

        while True:
            action = self.agent.get_action(obs, 0.1, self.hand)
            l = self.hand_run_val()
            if action not in l:
                return Card(action - 1)
            card = 0
            for x in self.hand:
                if x.run_val == action:
                    card = x
                    break
            # card = self.ask_for_play(previous_plays)  # subclasses (that is, actual players) must implement this
            # print("Nominated card", card)
            if sum((pp.value for pp in previous_plays)) + card.value < 32:
                self.update_after_play(card)
                return card
            else:
                # `self.ask_for_play` has returned a card that would put the
                # score above 31 but the player's hand contains at least one
                # card that could be legally played (you're not allowed to say
                # "go" here if you can legally play). How the code knows that
                # the player has a legal move is beyond me
                print('>>> You nominated', card,
                      'but that is not a legal play given your hand. You must play if you can')
                return "Go!"

    def ask_for_discards(self):
        """
        For each possible discard, score and select
        highest scoring move. It will discard cards
        that create the least value.
        """

        print("cribbage: {} is choosing discards".format(self))
        deck = Deck().draw(52)
        potential_cards = [n for n in deck if n not in self.hand]
        bar = tqdm(total=226994)
        discards = []
        mean_scores = []
        for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
            inner_scores = []
            for pot in combinations(potential_cards, 3):  # 46 choose 3 == 15,180
                inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]))
                bar.update(1)
            inner_scores = np.array(inner_scores)
            discards.append(discard)
            mean_scores.append(inner_scores.mean())

        return list(discards[np.argmin(mean_scores)])


class DQNAgentPlayer(Player):
    """
    "Expert systems" style AI player that systematically
    enumerates possible moves and chooses the move that
    maximizes its score after the move
    """

    def updateAgent(self):
        parameters = {
            "action_dimension": 15 * 1,
            "observation_dimension": 13 * 14,
            "hidden_layer_dimension": 256,
            "learning_rate": 0.1
        }
        self.agent = DQNAgent(parameters)

    def play(self, count, previous_plays, obs):
        """Public method"""
        if not self.hand:
            print('>>> I have no cards', self)
            return "No cards!"
        elif all(count + card.value > 31 for card in self.hand):
            print(">>>", self, self.hand, "I have to say 'go' on that one")
            return "Go!"

        while True:
            action = self.agent.get_action(obs, 0.1, self.hand)
            l = self.hand_run_val()
            if action not in l:
                return Card(action - 1)
            card = 0
            for x in self.hand:
                if x.run_val == action:
                    card = x
                    break
            # card = self.ask_for_play(previous_plays)  # subclasses (that is, actual players) must implement this
            # print("Nominated card", card)
            if sum((pp.value for pp in previous_plays)) + card.value < 32:
                self.update_after_play(card)
                return card
            else:
                # `self.ask_for_play` has returned a card that would put the
                # score above 31 but the player's hand contains at least one
                # card that could be legally played (you're not allowed to say
                # "go" here if you can legally play). How the code knows that
                # the player has a legal move is beyond me
                print('>>> You nominated', card,
                      'but that is not a legal play given your hand. You must play if you can')
                return "Go!"

    def ask_for_discards(self):
        """
        For each possible discard, score and select
        highest scoring move. It will discard cards
        that create the least value.
        """

        print("cribbage: {} is choosing discards".format(self))
        deck = Deck().draw(52)
        potential_cards = [n for n in deck if n not in self.hand]
        bar = tqdm(total=226994)
        discards = []
        mean_scores = []
        for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
            inner_scores = []
            for pot in combinations(potential_cards, 3):  # 46 choose 3 == 15,180
                inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]))
                bar.update(1)
            inner_scores = np.array(inner_scores)
            discards.append(discard)
            mean_scores.append(inner_scores.mean())

        return list(discards[np.argmin(mean_scores)])


class NFSPAgentPlayer(Player):
    """
    "Expert systems" style AI player that systematically
    enumerates possible moves and chooses the move that
    maximizes its score after the move
    """

    def updateAgent(self):
        parameters = {
            "action_dimension": 15 * 1,
            "observation_dimension": 13 * 14,
            "hidden_layer_dimension": 256,
            "learning_rate": 0.1
        }
        self.agent = NFSPPolicyAgent(parameters)

    def play(self, count, previous_plays, obs):
        """Public method"""
        if not self.hand:
            print('>>> I have no cards', self)
            return "No cards!", False
        elif all(count + card.value > 31 for card in self.hand):
            print(">>>", self, self.hand, "I have to say 'go' on that one")
            return "Go!", False

        while True:
            action, bestresponse = self.agent.get_action(obs, 0.1, self.hand)
            l = self.hand_run_val()
            if action not in l:
                print(action)
                return Card(action - 1), bestresponse
            card = 0
            for x in self.hand:
                if x.run_val == action:
                    card = x
                    break
            # card = self.ask_for_play(previous_plays)  # subclasses (that is, actual players) must implement this
            # print("Nominated card", card)
            if sum((pp.value for pp in previous_plays)) + card.value < 32:
                self.update_after_play(card)
                return card, bestresponse
            else:
                # `self.ask_for_play` has returned a card that would put the
                # score above 31 but the player's hand contains at least one
                # card that could be legally played (you're not allowed to say
                # "go" here if you can legally play). How the code knows that
                # the player has a legal move is beyond me
                print('>>> You nominated', card,
                      'but that is not a legal play given your hand. You must play if you can')
                return "Go!", bestresponse

    def ask_for_discards(self):
        """
        For each possible discard, score and select
        highest scoring move. It will discard cards
        that create the least value.
        """

        print("cribbage: {} is choosing discards".format(self))
        deck = Deck().draw(52)
        potential_cards = [n for n in deck if n not in self.hand]
        bar = tqdm(total=226994)
        discards = []
        mean_scores = []
        for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
            inner_scores = []
            for pot in combinations(potential_cards, 3):  # 46 choose 3 == 15,180
                inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]))
                bar.update(1)
            inner_scores = np.array(inner_scores)
            discards.append(discard)
            mean_scores.append(inner_scores.mean())

        return list(discards[np.argmin(mean_scores)])


class DDQNAgentPlayer(Player):
    """
    "Expert systems" style AI player that systematically
    enumerates possible moves and chooses the move that
    maximizes its score after the move
    """

    def updateAgent(self):
        parameters = {
            "action_dimension": 15 * 1,
            "observation_dimension": 13 * 14,
            "hidden_layer_dimension": 256,
            "learning_rate": 0.1
        }
        self.agent = DQNAgent(parameters)

    def play(self, count, previous_plays, obs):
        """Public method"""
        if not self.hand:
            print('>>> I have no cards', self)
            return "No cards!"
        elif all(count + card.value > 31 for card in self.hand):
            print(">>>", self, self.hand, "I have to say 'go' on that one")
            return "Go!"

        while True:
            action = self.agent.get_action(obs, 0.1, self.hand)
            l = self.hand_run_val()
            if action not in l:
                return Card(action - 1)
            card = 0
            for x in self.hand:
                if x.run_val == action:
                    card = x
                    break
            # card = self.ask_for_play(previous_plays)  # subclasses (that is, actual players) must implement this
            # print("Nominated card", card)
            if sum((pp.value for pp in previous_plays)) + card.value < 32:
                self.update_after_play(card)
                return card
            else:
                # `self.ask_for_play` has returned a card that would put the
                # score above 31 but the player's hand contains at least one
                # card that could be legally played (you're not allowed to say
                # "go" here if you can legally play). How the code knows that
                # the player has a legal move is beyond me
                print('>>> You nominated', card,
                      'but that is not a legal play given your hand. You must play if you can')
                return "Go!"

    def ask_for_discards(self):
        """
        For each possible discard, score and select
        highest scoring move. It will discard cards
        that create the least value.
        """

        print("cribbage: {} is choosing discards".format(self))
        deck = Deck().draw(52)
        potential_cards = [n for n in deck if n not in self.hand]
        bar = tqdm(total=226994)
        discards = []
        mean_scores = []
        for discard in combinations(self.hand, 2):  # 6 choose 2 == 15
            inner_scores = []
            for pot in combinations(potential_cards, 3):  # 46 choose 3 == 15,180
                inner_scores.append(score_hand([*discard, *pot[:-1]], pot[-1]))
                bar.update(1)
            inner_scores = np.array(inner_scores)
            discards.append(discard)
            mean_scores.append(inner_scores.mean())

        return list(discards[np.argmin(mean_scores)])


'''
        scores = []
        plays = []
        for card in self.hand:
            plays.append(card)
            scores.append(score_count(previous_plays + [card]))
        max_index = np.argmax(scores)

        return plays[max_index]
        '''

# class TrainedAIPlayer(Player):
#     """
#     A player that makes choices based on previous games
#     """

#     def __init__(self, name=""):
#         # override this constructor becasue I want to
#         # load the ML model in when we init AIPlayer instance
#         self.name = name
#         self.hand = []
#         self.score = 0
#         self.debug = False
#         self.model = load_trained_model()  # trained model we can ask directly for plays

#     def ask_for_input(self, play_vector):
#         card = self.model.ask_for_pegging_play(play_vector, self.in_hand)
#         card.ontable = True
#         return card

#     def ask_for_discards(self):
#         cards = self.model.ask_model_for_discard(
#             self.hand
#         )  # note: returns card objects
#         self.hand = [n for n in self.hand if n not in cards]
#         return cards


NondeterministicAIPlayer = RandomPlayer
