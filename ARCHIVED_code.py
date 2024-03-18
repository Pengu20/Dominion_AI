




# The debug buy action for domonion_game.py
    def __debug_buy_action(self, card_bought, players, main_player, players_input, game_history_file):
            game_history_file.write("----------- CARD bought -----------")
            index = sm.card_idx_2_set_idx(card_bought, self.game_state)

            Dominion_cards = self.game_state["dominion_cards"][index]
            game_history_file.write(f"Card: {Dominion_cards}")


            game_history_file.write(f"------------------- BEFORE -------------------")

            cards_in_discard = players[main_player]["cards_in_discard"]
            game_history_file.write(f"cards in discard: {cards_in_discard} \n")

            owned_cards = players[main_player]["owned_cards"]
            length_owned_cards = len(players[main_player]["owned_cards"])
            game_history_file.write(f"owned cards: {owned_cards} -> size -> {length_owned_cards}")
            
            buys = players[main_player]["buys"]
            game_history_file.write(f"buys: {buys} \n")

            value = players[main_player]["value"]
            game_history_file.write(f"player value: {value} \n")


            supply_amount = self.game_state["supply_amount"]
            game_history_file.write(f"card supply: {supply_amount} \n")
       



            main = int(main_player)
            advesary = int(main_player*(-1) + 1)
            card_effects().play_card(card_bought, self.game_state, players[main], players_input[main],  players[advesary], players_input[advesary])





            game_history_file.write(f"------------------- AFTER -------------------")
            cards_in_discard = players[main_player]["cards_in_discard"]
            game_history_file.write(f"cards in discard: {cards_in_discard} \n")


            owned_cards = players[main_player]["owned_cards"]
            length_owned_cards = len(players[main_player]["owned_cards"])
            game_history_file.write(f"owned cards: {owned_cards} -> size -> {length_owned_cards}")
            

            buys = players[main_player]["buys"]
            game_history_file.write(f"buys: {buys} \n")


            value = players[main_player]["value"]
            game_history_file.write(f"player value: {value} \n")


            supply_amount = self.game_state["supply_amount"]
            game_history_file.write(f"card supply: {supply_amount} \n")

