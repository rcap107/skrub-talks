#%%
import skrub
import skrub.selectors as s

data = skrub.datasets.fetch_credit_fraud()

baskets = skrub.var("baskets", data.baskets)
products = skrub.var("products", data.products)  # add a new variable

X = baskets[["ID"]].skb.mark_as_X()
y = baskets["fraud_flag"].skb.mark_as_y()
from skrub import selectors as s

vectorizer = skrub.TableVectorizer(high_cardinality=skrub.StringEncoder())
vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID").skb.set_name("product_vectorizer")
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()

features = X.merge(aggregated_products, left_on="ID", right_on="basket_ID").skb.set_description("Merging features with indices")
features = features.drop(columns=["ID", "basket_ID"])
from sklearn.ensemble import ExtraTreesClassifier

predictions = features.skb.apply(ExtraTreesClassifier(n_jobs=-1), y=y).skb.set_name("extra_trees_predictions")

# %%
predictions.skb.full_report(output_dir="resources/dataop_report")
# %%
