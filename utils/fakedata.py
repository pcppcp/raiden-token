import time
import numpy
import click
import random
import datetime

import logging
log = logging.getLogger(__name__)


def generate(kwargs):
    total_supply = kwargs['total_supply']
    bins = kwargs['bins']
    duration = kwargs['duration']

    price_start = kwargs['price_start']
    price_exponent = kwargs['price_exponent']
    price_constant = kwargs['price_constant']

    time_now = kwargs['start_time']

    max_bid = kwargs['max_bid']

    timestamped = [(time_now + i) for i in range(0, duration, 3600)]

    price_graph = []
    for i in timestamped:
        t = i - time_now
        price = price_start * ((1 + t) / (1 + t + ((t ** price_exponent) / price_constant)))
        price_graph.append(price)

    # probability density function
    pdf = lambda x, sigma, mu: (numpy.exp(-(numpy.log(x) - mu)**2 / (2 * sigma**2)) /
                                (x * sigma * numpy.sqrt(2 * numpy.pi)))

    bids = []
    funding_target = []
    sigma = kwargs['sigma']
    mu = kwargs['mu']
    bids_sum = 0
    auctioned = 0.5
    for current_price in price_graph:
        current_target = current_price * total_supply * auctioned
        log.info("%s price=%.3e sum=%.3e target=%.3e" % (
            datetime.timedelta(seconds=timestamped[price_graph.index(current_price)] - time_now),
            current_price, bids_sum, current_target))
        bid = max_bid * pdf(price_graph.index(current_price) / len(price_graph) * 2.5 + 0.01,
                            sigma, mu) * random.random()
        bids_sum += bid
        bids.append(bid)
        funding_target.append(current_target)
        if bids_sum >= current_target:
            break
    log.info("bids sum=%.3e, current target=%.3e" % (bids_sum, current_target))

    ar, ar_bins = numpy.histogram(timestamped[:len(bids)],
                                  bins=bins,
                                  weights=bids)
    price_ar = numpy.interp(numpy.arange(0, len(price_graph), len(price_graph) / bins),
                            numpy.arange(0, len(price_graph)), price_graph)

    target_ar = numpy.interp(numpy.arange(0, len(funding_target), len(funding_target) / bins),
                             numpy.arange(0, len(funding_target)), funding_target)
    return {
        'timestamped_bins': [int(x) for x in ar_bins[:-1].tolist()],
        'bin_sum': [int(x) for x in ar.tolist()],
        'bin_cumulative_sum': [int(x) for x in numpy.cumsum(ar).tolist()],
        'funding_target': target_ar.tolist(),
        'price': price_ar.tolist()
    }


def plot(data):
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    import datetime as dt

    def adjust_xaxis():
        ax = plt.gca()
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        plt.subplots_adjust(bottom=0.2)
        plt.xticks(rotation=25)

    def remove_xticks():
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_xticks([])
    plt.figure(1)

    # plotting price graph
    plt.subplot(411)
    plt.ylabel('price')
    remove_xticks()
    dates = [dt.datetime.fromtimestamp(ts) for ts in data['timestamped_bins']]
    plt.plot(dates, data['price'], 'r')

    # plotting  sum
    plt.subplot(412)
    remove_xticks()
    plt.ylabel('bids')
    plt.bar(dates, data['bin_sum'], 0.1)

    # plotting funding target
    plt.subplot(413)
    remove_xticks()
    plt.ylabel('target [Eth]')
    plt.plot(dates, data['funding_target'], 'r')

    # plotting cumulative sum
    plt.subplot(414)
    adjust_xaxis()
    plt.ylabel('bids total')
    plt.plot(dates, data['bin_cumulative_sum'], 'r')
    plt.show()


@click.command()
@click.option(
    '--sigma',
    default=1,
    type=float,
    help='sigma for the bid distribution function'
)
@click.option(
    '--mu',
    default=0,
    type=float,
    help='mu for the bid distribution function'
)
@click.option(
    '--total-supply',
    default=100e6,
    type=int,
    help='total token supply (tokens)'
)
@click.option(
    '--bins',
    default=800,
    type=int,
    help='bins in the output'
)
@click.option(
    '--duration',
    default=10 * 24 * 60 * 60,
    type=int,
    help='duration of the auction (seconds)'
)
@click.option(
    '--price-start',
    default=2 * 10**18,
    type=int,
    help='price start (wei)'
)
@click.option(
    '--price-exponent',
    default=3,
    type=float,
    help='price exponent'
)
@click.option(
    '--price-constant',
    default=1574640000,
    type=int,
    help='price constant'
)
@click.option(
    '--start-time',
    default=time.time(),
    type=int,
    help='price constant'
)
@click.option(
    '--max-bid',
    default=1e10,
    type=int,
    help='bid cap'
)
@click.option(
    '--plot',
    is_flag=True,
    default=False,
    help='display result (requires matplotlib)'
)
@click.option(
    '--json',
    is_flag=True,
    default=False,
    help='print result as a JSON to stdout'
)
def main(**kwargs):
    if kwargs['json']:
        log.setLevel(level=logging.FATAL)
    data = generate(kwargs)
    if kwargs['plot']:
        plot(data)
    if kwargs['json']:
        import json
        print(json.dumps(data))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
