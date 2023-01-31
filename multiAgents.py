"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name: Reut Dayan
Student ID: 206433245

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)


import random, util, math

from connect4 import Agent


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 1  # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):

        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.isWin():
        Returns whether or not the game state is a winning state for the current turn player

        gameState.isLose():
        Returns whether or not the game state is a losing state for the current turn player

        gameState.is_terminal()
        Return whether or not that state is terminal
        """

        return self.min_max_decision(gameState)

    def min_max_decision(self, gameState):
        action_values = {}
        for action in gameState.getLegalActions(self.index):
            successor_gameState = gameState.generateSuccessor(self.index, action)
            action_values[action] = self.min_max_value(successor_gameState, self.depth - 1, False)
        return max(action_values, key=action_values.get)

    def min_max_value(self, gameState, depth, is_max):
        gameState.turn = int(is_max)
        if gameState.is_terminal():
            return self.evaluationFunction(gameState)
        if depth == 0:
            return self.evaluationFunction(gameState)

        # agent turn
        if is_max:
            max_successor_evaluate_value = -math.inf
            for new_action in gameState.getLegalActions(self.index):
                successor_gameState = gameState.generateSuccessor(self.index, new_action)
                max_successor_evaluate_value = max(max_successor_evaluate_value,
                                                   self.min_max_value(successor_gameState, depth - 1, False))
            return max_successor_evaluate_value

        # opponent turn
        else:
            min_successor_evaluate_value = math.inf
            for new_action in gameState.getLegalActions(self.index):
                successor_gameState = gameState.generateSuccessor(self.index, new_action)
                min_successor_evaluate_value = min(min_successor_evaluate_value,
                                                   self.min_max_value(successor_gameState, depth - 1, True))
            return min_successor_evaluate_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return self.alpha_beta_search(gameState, self.depth)

    def alpha_beta_search(self, gameState, depth):
        v = -math.inf
        alpha = -math.inf
        beta = math.inf
        action_target = None
        for action in gameState.getLegalActions(self.index):
            successor_gameState = gameState.generateSuccessor(self.index, action)
            eval_value = self.min_value(successor_gameState, depth - 1, alpha, beta)
            if eval_value > v:
                action_target = action
                v = eval_value
            if v > beta:
                return action_target  # beta cutoff
            alpha = max(alpha, v)
        return action_target

    def max_value(self, gameState, depth, alpha, beta):
        gameState.turn = 1
        if gameState.is_terminal():
            return self.evaluationFunction(gameState)
        if depth == 0:
            return self.evaluationFunction(gameState)

        v = -math.inf
        for action in gameState.getLegalActions(self.index):
            successor_gameState = gameState.generateSuccessor(self.index, action)
            v = max(v, self.min_value(successor_gameState, depth - 1, alpha, beta))
            if v > beta:
                return v  # beta cutoff
            alpha = max(alpha, v)
        return v

    def min_value(self, gameState, depth, alpha, beta):
        gameState.turn = 0
        if gameState.is_terminal():
            return self.evaluationFunction(gameState)
        if depth == 0:
            return self.evaluationFunction(gameState)

        v = math.inf
        for action in gameState.getLegalActions(self.index):
            successor_gameState = gameState.generateSuccessor(self.index, action)
            v = min(v, self.max_value(successor_gameState, depth - 1, alpha, beta))
            if v < alpha:
                return v  # alpha cutoff
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        v = -math.inf
        action_target = None
        for action in gameState.getLegalActions(self.index):
            successor_gameState = gameState.generateSuccessor(self.index, action)
            eval_value = self.expectimax_value(successor_gameState, self.depth - 1, False)
            if eval_value > v:
                action_target = action
            v = max(v, eval_value)
        return action_target

    def expectimax_value(self, gameState, depth, is_agent):
        gameState.turn = int(is_agent)
        if gameState.is_terminal():
            return self.evaluationFunction(gameState)
        if depth == 0:
            return self.evaluationFunction(gameState)

        if is_agent:  # Agent turn
            v = -math.inf
            for action in gameState.getLegalActions(self.index):
                successor_gameState = gameState.generateSuccessor(self.index, action)
                v = max(v, self.expectimax_value(successor_gameState, depth - 1, False))
        else:  # Opponent turn
            v = 0
            for action in gameState.getLegalActions(self.index):
                successor_gameState = gameState.generateSuccessor(self.index, action)
                p = 1 / len(gameState.getLegalActions(self.index))  # Probability of each action
                v += p * self.expectimax_value(successor_gameState, depth - 1, True)
        return v
