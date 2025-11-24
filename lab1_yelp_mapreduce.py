import json
from collections import defaultdict

BUSINESS_PATH = "yelp_academic_dataset_business.json"

MIN_BUSINESSES_PER_CITY = 10


def mapper(line):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return []

    city = obj.get("city")
    stars = obj.get("stars")

    if not city or stars is None:
        return []

    city = str(city).strip()
    try:
        stars = float(stars)
    except (TypeError, ValueError):
        return []

    return [(city, (stars, 1))]


def shuffle_and_sort(mapped_items):
    groups = defaultdict(list)
    for key, value in mapped_items:
        groups[key].append(value)
    return groups


def reducer(city, values):
    sum_stars = 0.0
    count = 0

    for stars, n in values:
        sum_stars += stars
        count += n

    if count == 0:
        return None

    avg = sum_stars / count
    return city, avg, count


def run_mapreduce(path):
    mapped = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kv_pairs = mapper(line)
            mapped.extend(kv_pairs)

    grouped = shuffle_and_sort(mapped)

    city_stats = []

    for city, values in grouped.items():
        result = reducer(city, values)
        if result is None:
            continue

        city, avg, count = result

        if count >= MIN_BUSINESSES_PER_CITY:
            city_stats.append((city, avg, count))

    best_city = None
    best_avg = -1.0
    best_count = 0

    for city, avg, count in city_stats:
        if avg > best_avg:
            best_city = city
            best_avg = avg
            best_count = count

    return best_city, best_avg, best_count, city_stats


def main():
    best_city, best_avg, best_count, city_stats = run_mapreduce(BUSINESS_PATH)

    print("Город с максимальным средним рейтингом заведений "
          f"(учитываются только города с N >= {MIN_BUSINESSES_PER_CITY} заведений):")
    print(f"{best_city} — средний рейтинг {best_avg:.3f}, количество заведений: {best_count}")

    city_stats_sorted = sorted(city_stats, key=lambda x: x[1], reverse=True)

    print("\nТоп-10 городов по среднему рейтингу:")
    for i, (city, avg, count) in enumerate(city_stats_sorted[:10], start=1):
        print(f"{i:2d}. {city:25s} | средний рейтинг = {avg:.3f} | заведений = {count}")


if __name__ == "__main__":
    main()
