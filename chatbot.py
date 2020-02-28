# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens

import numpy as np
import re

from PorterStemmer import PorterStemmer
stemmer = PorterStemmer()


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'VAHN'
        self.creative = creative
        self.titles, ratings = movielens.ratings()
        self.sentiment = movielens.sentiment()

        self.sentiment_stemmed = {}
        for word in self.sentiment:
            self.sentiment_stemmed[stemmer.stem(word)] = self.sentiment[word]
        self.debuggy = []

        #to help print states
        self.recommendations = 0
        self.clarify = False
        self.candidates = []
        self.originalLine = ""
        self.moreInfo = False
        self.movieTitle = ""
        self.multipleMovies = False

        #############################################################################
        # TODO: Binarize the movie ratings matrix.                                  #
        #############################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        #############################################################################

        greeting_message = "Hello! My name is VAHN. I want to recommend a movie to you but first I need to ask you about your movie preferences. Tell me about a movie you liked or disliked."

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        #############################################################################

        goodbye_message = "Thank you for chatting with me. Have a nice day!"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return goodbye_message

    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method,         #
        # possibly calling other functions. Although modular code is not graded,    #
        # it is highly recommended.                                                 #
        #############################################################################
        response = ""
        if self.creative:
            return self.process_starter(line)
            #response = "I processed {} in creative mode!".format(line)

        #STARTER RESPONSES
        else:
            return self.process_starter(line)

    def process_starter(self, line):
        text = ""
        foundTitle = False

        #OPTION 1: CLARIFICATION -- if there are multiple movie options
        if self.clarify == True:
            dis_titles = self.disambiguate(line, self.candidates)
            if len(dis_titles) == 1:
                self.clarify = False
                self.candidates = []
                text = self.titles[dis_titles[0]][0]
                foundTitle = True
            else:
                self.candidates = dis_titles
                r = "Please provide more clarification about which movie it is (i.e. unique word): "
                for i in range(len(dis_titles)):
                    r += self.titles[dis_titles[i]][0]
                    if i < (len(dis_titles)-1):
                        r += ", "
                    else:
                        r += ":"
                return r

        #OPTION 2: NARROWED IT DOWN TO ONE MINUTE
        else:
            self.originalLine = line

            #SENTIMENT ClARIFY: check if the user if providing more info for sentiment
            if self.moreInfo:
                text = self.movieTitle
                self.moreInfo = False

            #INPUT OF MOVIE
            else:
                print(line)
                text = self.extract_titles(line)
                movies = []
                for t in text:
                    potential_titles = self.find_movies_by_title(t)
                    #EDIT: NEED TO CHECK IF NONE OF THEM ARE REAL TITLES
                    movies.append(potential_titles)

                # ADDRESS MULTIPLE MOVIES, for now just looks at first one
                if len(movies) >= 1:
                    if len(movies[0]) > 1:
                        self.candidates = movies[0]
                        self.clarify = True
                        r = "Please provide more clarification about which movie it is (i.e. unique word): "
                        for i in range(len(movies[0])):
                            r += self.titles[movies[0][i]][0]
                            if i < (len(movies[0])-1):
                                r += ", "
                            else:
                                r += ":"
                        return r
                else:
                    return "I'm sorry. We could not find any titles by that name. Please enter another title."

        #text = self.titles[movies[0][0]][0]
        sentiment = self.extract_sentiment(self.originalLine)

        sent = ""
        if sentiment == 1:
            sent = "You liked \"{}\"! ".format(text)
        elif sentiment == -1:
            sent = "You did not like \"{}\"! ".format(text)
        else:
            sent = "I'm sorry, I am not sure how you felt about \"{}\". Could you provide me with more information? ".format(text)
            self.moreInfo = True
            self.movieTitle = text
            return sent

        if self.recommendations < 5:
            response = sent + "Please tell me your thoughts on another movie."
            self.recommendations += 1
        else:
            self.recommendations = 0
            recommend = "That is enough for me to make a recommendation. I recommend that you watch \"XX\"!" #.format(recommend)
            optionToQuit = "Would you like me to give you another recommendation? If YES enter Y, and if NO enter :quit if you're done"
            response = sent + recommend + optionToQuit
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information from a line of text.

        Given an input line of text, this method should do any general pre-processing and return the
        pre-processed string. The outputs of this method will be used as inputs (instead of the original
        raw text) for the extract_titles, extract_sentiment, and extract_sentiment_for_movies methods.

        Note that this method is intentially made static, as you shouldn't need to use any
        attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        #############################################################################
        # TODO: Preprocess the text into a desired format.                          #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to your    #
        # implementation to do any generic preprocessing, feel free to leave this   #
        # method unmodified.                                                        #
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess('I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        #CREATIVE MODE
        if self.creative:
            movies = []

            for title in self.titles:
                formatted_title = title[0].lower()
                formatted_title = re.sub(' \(.*\)', '', formatted_title)

                if re.search(', the$', formatted_title):
                    formatted_title = re.sub(', the$', '', formatted_title)
                    formatted_title = 'the ' + formatted_title

                movie_name = formatted_title

                formatted_title = re.sub('\*', '\*', formatted_title)
                formatted_title = re.sub('\+', '\+', formatted_title)
                formatted_title = re.sub('\$', '\$', formatted_title)
                formatted_title = '(^|[ ,\"])' + formatted_title + '([ ,!?;:\.\"]|$)'

                if  re.search(formatted_title, preprocessed_input, re.IGNORECASE):
                    movies.append(movie_name),
            print(movies)
            return movies

        #STARTER MODE
        else:
            quote_indices = [i for i, ltr in enumerate(preprocessed_input) if ltr == '\"']
            movie_titles = []
            for i in range(0, len(quote_indices), 2):
                    movie_titles.append(preprocessed_input[quote_indices[i] + 1:quote_indices[i+1]])
        return movie_titles


    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 1953]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        # done
        movies = []
        title = title.lower().split()
        for i in range(len(self.titles)):
            present = True
            for j in title:
                index = (self.titles[i][0].lower()).find(j)
                if index == -1:
                    present = False
            if present:
                movies.append(i)
        return movies

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the text
        is super negative and +2 if the sentiment of the text is super positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess('I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        preprocessed_input = preprocessed_input.replace('\'', '')
        words = preprocessed_input.split()
        negated = False
        doubled = False
        sentiment = 0
        double_neg_count = 0
        double_pos_count = 0
        neg = '(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)'
        double = '(?:ve+r+y+|really|re+a+l+y+|to+tal+y+|absolu+tel+y+|tru+l+y+|extre+mel+y+)'
        strong_pos = '(?:great|awesome+|per+fect|ama+zing|excellent|fantastic|exceptional|ter+ific|wonderful|marvelous|lo+ve+d??)'
        strong_neg = '(?:hated|ter+ible|hor+ible|aw+ful|appaling|atrocious|dreadful|ha+te+d?|abhorr?e?d?|despised?)'
        #go through all the input words
        for word in words:
            sent = 0
            stemmed = stemmer.stem(word)

            if re.match(double, word):
                doubled = True

            #check if strong positive word
            elif re.match(strong_pos, word):
                print("strong pos")
                double_pos_count += 1
                if doubled:
                    sent = 4
                else:
                    sent = 2
                doubled = False

            #check if strong negative word
            elif re.match(strong_neg, word):
                double_neg_count += 1
                if doubled:
                    sent = -4
                else:
                    sent = -2
                doubled = False

            elif stemmed in self.sentiment_stemmed:
                if self.sentiment_stemmed[stemmed] == 'pos':
                    sent = 1
                    if doubled == True:
                        double_pos_count += 1
                        sent *= 2
                        doubled = False
                else:
                    sent = -1
                    if doubled == True:
                        double_neg_count += 1
                        sent *= 2
                        doubled = False

            if negated == True:
                sent *= -1
            sentiment += sent

            #turn on and turn off negation
            if re.match(neg, word):
                negated = True
            elif re.match('^[.:;!?]$', word):
                negated = False

        if self.creative:
            if sentiment > 1:
                if double_pos_count > double_neg_count:
                    sentiment = 2
                else:
                    sentiment = 1
            elif sentiment < -1:
                if double_pos_count < double_neg_count:
                    sentiment = -2
                else:
                    sentiment = -1
        else:
            if sentiment > 1:
                sentiment = 1
            elif sentiment < -1:
                sentiment = -1

        return sentiment


    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of pre-processed text
        that may contain multiple movies. Note that the sentiments toward
        the movies may be different.

        You should use the same sentiment values as extract_sentiment, described above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess('I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie title,
          and the second is the sentiment in the text toward that movie
        """
        titles = self.extract_titles(preprocessed_input)
        titlemap = {}

        new_input = preprocessed_input

        for i in range(len(titles)):
            new_input = re.sub(titles[i], 'movie%s' % i, new_input.lower())
            titlemap['movie%s' % i] = titles[i]


        clauses = re.split('(but|however|\.|\!|,)', new_input)
        to_remove = ['but', 'however', '.', '!', ',']
        for rem in to_remove:
            if rem in clauses:
                clauses.remove(rem)

        sentiments = []
        for i in range(len(clauses)-1, -1, -1):
            matches = re.findall('movie[\d]+', clauses[i])

            if len(titles) == 0:
                clauses[i-1] = clauses[i-1] + clauses[i]
            else:
                if 'not' in clauses[i]:
                    sentiment = -1 * self.extract_sentiment(clauses[i-1])
                else:
                    sentiment = self.extract_sentiment(clauses[i])
                for match in matches:
                    sentiments.append((re.findall(titlemap[match], preprocessed_input, re.IGNORECASE)[0], sentiment))
        return sentiments


    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least edit distance
        from the provided title, and with edit distance at most max_distance.

        - If no movies have titles within max_distance of the provided title, return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given title
          than all other movies, return a 1-element list containing its index.
        - If there is a tie for closest movie, return a list with the indices of all movies
          tying for minimum edit distance to the given movie.

        Example:
          chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
        """
        def get_edit_distance(title1, title2):
            n = len(title1)
            m = len(title2)
            d = [[0 for i in range(m + 1)] for j in range(n + 1)]
            for i in range(n+1):
                d[i][0] = i
            for i in range(m+1):
                d[0][i] = i
            for i in range(1,n+1):
                for j in range(1,m+1):
                    if title1[i-1].lower() == title2[j-1].lower():
                        d[i][j] = min(d[i-1][j] + 1, min(d[i][j-1] + 1, d[i-1][j-1]))
                    else:
                        d[i][j] = min(d[i-1][j] + 1, min(d[i][j-1] + 1, d[i-1][j-1] + 2))
            return d[n][m]

        least_edit_distance = float('inf')
        edit_distance_map = {}
        for i in range(len(self.titles)):
            movie = self.titles[i][0]
            movie = movie[:movie.rfind('(')]
            edit_distance = get_edit_distance(movie, title)
            if edit_distance <= max_distance and edit_distance <= least_edit_distance:
                least_edit_distance = edit_distance
                if edit_distance not in edit_distance_map:
                    edit_distance_map[edit_distance] = []
                edit_distance_map[edit_distance].append(i)

        return edit_distance_map[least_edit_distance] if least_edit_distance != float('inf') else []


    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be talking about
        (represented as indices), and a string given by the user as clarification
        (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
        or Titanic (1997)?"), use the clarification to narrow down the list and return
        a smaller list of candidates (hopefully just 1!)

        - If the clarification uniquely identifies one of the movies, this should return a 1-element
        list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it should return a list
        with the indices it could be referring to (to continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by the clarification
        """
        #loop through all the candidates to find max substring of words
        counts = [0]*len(candidates)
        for i in range(len(candidates)):

            #format the name of the movie
            movie = self.titles[candidates[i]][0].lower()
            if re.search(', the$', movie):
                movie = re.sub(', the$', '', movie)
                movie = 'the ' + movie

            #find largest substring of full words
            a = movie.split()
            for w in range(len(a)):
                word = re.sub('\(', '', a[w])
                word = re.sub("\)", "", word)
                a[w] = word
            b = clarification.split()

            #longest common subsequence
            i_m = 0
            i_c = 0
            max_count = 0
            curr_count = 0
            while True:
                if a[i_m] == b[i_c]:
                    curr_count += 1
                    i_m += 1
                    i_c += 1
                else:
                    if curr_count > max_count:
                        max_count = curr_count
                    curr_count = 0
                    i_c += 1
                #break if done with movie words & reset if
                if i_c == len(b):
                    i_m +=1
                    i_c = 0
                if i_m >= len(a):
                    break

            if curr_count > max_count:
                max_count = curr_count
            counts[i] = max_count

        #Don't identify by movie title but other inputs
        if max(counts) == 0:
            if re.match(clarification, "^ *\d[\d ]*$"):
                i = int(clarification)
                if i <= len(candidates):
                    return candidates[i-1]
            if re.match(clarification, "most recent"):
                return oldest
            if re.match(clarification, "most recent"):
                return recent

        #prints the longest sub-sequene
        identified = [i for i, j in enumerate(counts) if j == max(counts)]
        final = []
        for i in identified:
            final.append(candidates[i])
        return final

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use any
        attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered positive

        :returns: a binarized version of the movie-rating matrix
        """
        #############################################################################
        # TODO: Binarize the supplied ratings matrix. Do not use the self.ratings   #
        # matrix directly in this function.                                         #
        #############################################################################

        # The starter code returns a new matrix shaped like ratings but full of zeros.
        for i in range(len(ratings)):
            for j in range(len(ratings[i])):
                if ratings[i][j] > threshold:
                    ratings[i][j] = 1
                elif ratings[i][j] <= threshold and ratings[i][j] > 0:
                    ratings[i][j] = -1
        return ratings

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        #############################################################################
        # TODO: Compute cosine similarity between the two vectors.
        #############################################################################
        similarity = np.inner(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in ratings_matrix,
          in descending order of recommendation
        """

        #######################################################################################
        # TODO: Implement a recommendation function that takes a vector user_ratings          #
        # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
        # Do not use the self.ratings matrix directly in this function.                       #
        #                                                                                     #
        # For starter mode, you should use item-item collaborative filtering                  #
        # with cosine similarity, no mean-centering, and no normalization of scores.          #
        #######################################################################################

        # Populate this list with k movie indices to recommend to the user.
        if not creative:
            recommendations = {} # dict of index to estimated rating
            rated = (user_ratings != 0) # get indices of nonzero ratings
            rated_indices = np.where(rated)[0]

            for j in range(len(ratings_matrix)):
                if j not in rated_indices and sum(ratings_matrix[j]) != 0:
                        est_rating = 0
                        for index in rated_indices:
                            est_rating += self.similarity(ratings_matrix[index], ratings_matrix[j])*user_ratings[index]
                        recommendations[j] = est_rating
        # sort by estimated rating, get 1st k and then get indices
        final_recs = sorted(recommendations.items(), key=lambda item: item[1], reverse = True)
        final_recs = final_recs[:k]
        final_recs = [x[0] for x in final_recs]

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return final_recs

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
        """Return debug information as a string for the line string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = str(self.debuggy)
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your chatbot
        can do and how the user can interact with it.
        """
        intro = "Hello! My name is VAHN and I specialize in movie preferences. "
        return intro
        # Your task is to implement the chatbot as detailed in the PA6 instructions.
        # Remember: in the starter mode, movie names will come in quotation marks and
        # expressions of sentiment will be simple!
        # Write here the description for your own chatbot!
        # """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')

