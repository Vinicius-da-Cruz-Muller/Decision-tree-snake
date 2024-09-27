import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score  # Importando as métricas
from helper import plot

# random.seed(42)
# np.random.seed(42)

MAX_MEMORY = 100_000
BATCH_SIZE = 950

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 50#fator de aleatoriedade (exploração)
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # Modificando hiperparâmetros da árvore de decisão
        self.model = DecisionTreeClassifier(
            max_depth=30,             # Aumenta a profundidade máxima
            min_samples_split=5,     # Exige mais amostras para dividir um nó
            min_samples_leaf=1,       # Cada folha deve ter no mínimo x amostras
            criterion='entropy'       # Usa o ganho de informação em vez do Gini
        )
        self.is_trained = False  # Verificar se o modelo já foi treinado

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.y, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Perigo indo reto
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Perigo à direita
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Perigo à esquerda
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Movimentos atuais
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Posição da comida
            game.food.x < game.head.x,  # Comida à esquerda
            game.food.x > game.head.x,  # Comida à direita
            game.food.y < game.head.y,  # Comida acima
            game.food.y > game.head.y  # Comida abaixo
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # Preparar os dados para o treinamento
        X = np.array(states)  # Características
        y = np.argmax(actions, axis=1)  # Labels

        self.model.fit(X, y)
        self.is_trained = True

    def get_action(self, state):
        # Exploração vs Exploração
        self.epsilon = max(0.01, 50 * np.exp(-0.001 * self.n_games))  # Decaimento exponencial
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon or not self.is_trained:
            # Escolha aleatória para explorar
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Usar a árvore de decisão treinada
            prediction = self.model.predict([state])
            move = prediction[0]
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    # Armazenando as ações reais e preditas
    y_true = []
    y_pred = []

    while True:
        # Obter estado atual
        state_old = agent.get_state(game)

        # Obter ação
        final_move = agent.get_action(state_old)

        # Realizar ação e obter o novo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Memorizar a transição
        agent.remember(state_old, final_move, reward, state_new, done)

        # Armazenar ação real e prevista
        y_true.append(np.argmax(final_move))  # Ação real
        y_pred.append(np.argmax(agent.get_action(state_old)))  # Ação prevista pelo modelo

        if done:
            # Treinar a memória e plote os resultados
            game.reset()
            agent.n_games += 1
            agent.train()

            if score > record:
                record = score
            print('Jogo', agent.n_games, 'Score', score, 'Recorde:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Calcular e imprimir as métricas ao final de vários jogos
            if agent.n_games % 10 == 0:  # A cada 10 jogos, imprima as métricas
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

                print(f'Métricas após {agent.n_games} jogos:')
                print(f'Acurácia: {accuracy:.5f}, Precisão: {precision:.5f}, Recall: {recall:.5f}')

if __name__ == '__main__':
    train()
