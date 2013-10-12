package edu.umn.cs.recsys.uu;

import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.dao.ItemEventDAO;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.List;

/**
 * User-user item scorer.
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SimpleUserUserItemScorer extends AbstractItemScorer {
    private static final Logger logger = LoggerFactory.getLogger(SimpleUserUserItemScorer.class);

    private final UserEventDAO userDao;
    private final ItemEventDAO itemDao;

    @Inject
    public SimpleUserUserItemScorer(UserEventDAO udao, ItemEventDAO idao) {
        userDao = udao;
        itemDao = idao;
    }

    @Override
    public void score(long user, @Nonnull MutableSparseVector scores) {
        SparseVector userVector = getUserRatingVector(user);

        // TODO Score items for this user using user-user collaborative filtering

         CosineVectorSimilarity cosSimilarity = new CosineVectorSimilarity();


        //User's mean rating mu_u
        double userMeanRating = userVector.mean();
        MutableSparseVector userMeanVector = userVector.mutableCopy();
        // Compute mean-centered ratings
        userMeanVector.add( userMeanRating * -1.0);


        // This is the loop structure to iterate over items to score
        for (VectorEntry e: scores.fast(VectorEntry.State.EITHER)) {
            // item (movie ID)
            long i = e.getKey();
            logger.info("item {}", i);

            double sum = 0;
            double weight = 0;


            //Get all users who have rated the item
            LongSet potentialNeighbors =  itemDao.getUsersForItem(i);

            MutableSparseVector neighborSimilarities =  MutableSparseVector.create(potentialNeighbors);
            MutableSparseVector neighborMeanItemRatings =  MutableSparseVector.create(potentialNeighbors);

            neighborSimilarities.clear();
            neighborMeanItemRatings.clear();


            for (long v : potentialNeighbors){
                if( v != user){
                   MutableSparseVector neighborVector = getUserRatingVector(v).mutableCopy();

                    // Neighbor's mean rating mu_v
                    double neighborMeanRating = neighborVector.mean();

                    // Compute mean-centered ratings
                    neighborVector.add(neighborMeanRating * -1.0);

                    // Neighbor's item rating r_v,i
                    double neighborItemRating = neighborVector.get(i);

                    double vOffset = neighborItemRating - neighborMeanRating;

                    double sim = cosSimilarity.similarity(userMeanVector, neighborVector);
                    neighborSimilarities.set(v, sim);
                    neighborMeanItemRatings.set(v, vOffset);
                   // logger.info("Potential neighbor: user id = {} cosineSimilarity = {}", v, sim);
                }

            }
            // Sort the neighbors for the similarity value and then use the first 30
            LongArrayList neighbors = neighborSimilarities.keysByValue(true);

            for (long v : neighbors.subList(0, 30)){
                double sim = neighborSimilarities.get(v);
                double vOffset = neighborMeanItemRatings.get(v);
                logger.info("Neighbor: user id = {} cosineSimilarity = {} vOffset = {}", v, sim, vOffset);
                sum += sim * vOffset;
                weight += Math.abs(sim);
            }

            scores.set(e, userMeanRating +  sum / weight);
        }
    }

    /**
     * Get a user's rating vector.
     * @param user The user ID.
     * @return The rating vector.
     */
    private SparseVector getUserRatingVector(long user) {
        UserHistory<Rating> history = userDao.getEventsForUser(user, Rating.class);
        if (history == null) {
            history = History.forUser(user);
        }
        return RatingVectorUserHistorySummarizer.makeRatingVector(history);
    }
}
