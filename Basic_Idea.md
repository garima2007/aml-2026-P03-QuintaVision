# Hotel Booking Cancellation & Overbooking Decision
Hotels often face booking cancellations, which can result in empty rooms and loss of revenue. At the same time, accepting too many extra bookings can create an overbooking problem if more customers arrive than expected.
The basic idea of this project is to use historical hotel booking data to understand customer booking and cancellation patterns.
The project will aim to:
- Analyze hotel booking data and cancellation behaviour.
- Identify factors that may be related to booking cancellations.
- Build machine learning models to predict whether a booking is likely to be cancelled.
- Estimate the probability of cancellation for individual bookings.
- Compare different machine learning algorithms and select a suitable model.
- Use predicted cancellation probabilities to support an overbooking decision.
The target variable is `is_canceled`:
- `0` → Booking was not cancelled
- `1` → Booking was cancelled
The final goal is not only to predict cancellations, but also to explore how these predictions can help hotels reduce losses caused by empty rooms while controlling the risk of excessive overbooking.

In this project, we are working with historical hotel booking data to predict whether a booking is likely to be cancelled.
First, the dataset is analyzed and cleaned by handling missing values, checking invalid records, and removing features that may cause data leakage. Exploratory Data Analysis (EDA) is then performed to understand how factors such as lead time, hotel type, deposit type, market segment, customer type, and previous cancellations are related to booking cancellations.
After understanding the data, useful features are created and the dataset is prepared for machine learning using train-test splitting, categorical encoding, and numerical scaling.
Multiple machine learning classification models will be trained and compared to find a suitable model for predicting booking cancellations.
The selected model will then be used to estimate cancellation probabilities. These probabilities will be used to explore an overbooking strategy that aims to reduce losses from empty rooms while also considering the risk of accepting too many bookings.
