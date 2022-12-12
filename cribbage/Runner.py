from cribbage.game import Game
from cribbage.card import Card
import matplotlib.pyplot as plt
from cribbage.players import (
    WinGame,
    HumanPlayer,
    NondeterministicAIPlayer,
    GreedyAgentPlayer,
    DDQNAgentPlayer,
    DRQNAgentPlayer,
    DQNAgentPlayer,
    NFSPAgentPlayer
)


def main():
    dqn = DQNAgentPlayer('DQN Agent')
    ddqn = DDQNAgentPlayer('DDQN Agent')
    drqn = DRQNAgentPlayer('DRQN Agent')
    greedy = GreedyAgentPlayer('Greedy Agent')
    random = NondeterministicAIPlayer('Random Agent1')
    nfsp1 = NFSPAgentPlayer('NFSP Agent')
    nfsp2 = NFSPAgentPlayer('NFSP Agent')

    # Play game
    Player1 = random
    Player2 = dqn
    data = []
    selfPlay= False
    try:
        if selfPlay:
            game = Game(nfsp1, nfsp2)
            r = game.runSelfPlay(selfPlay, data)
        else:
            game = Game(Player1, Player2)
            r = game.runfinal(data)

        plt.plot(r)
        plt.ylabel('some numbers')
        plt.grid(True)
        plt.show()
        print(len(data))
    except WinGame as win_game:
        print(win_game)


if __name__ == '__main__':
    main()
