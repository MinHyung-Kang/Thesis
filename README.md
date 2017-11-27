# Speedup Methods for Stein Variational Gradient Descent

## Abstract

We introduce several variations of recently introduced Stein Variational Gradient Descent algorithm that reduces the original quadratic runtime complexity to linear time. Our models approximate the update for each particle by either looking at subset of original particles or redefining kernel values using another set of reference points which we name as induced points. We utilize various variance reduction techniques and introduce novel adversarial updates which counter-intuitively increases the discrepancy between particles to gain better updates. The models are tested on various datasets, and results suggest that compared to the original algorithm our models are not only faster but also often show better performance.

## Acknowledgments

This work could not have been done without my advisor, Professor [Qiang Liu](http://www.cs.dartmouth.edu/~qliu/) 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details