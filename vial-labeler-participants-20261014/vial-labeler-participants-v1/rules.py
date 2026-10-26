"""Inspection semantics, independent of the image classifier.

Findings are three-valued: true, false, or None (not established).
Vial appearance never establishes an upstream reject's validity.
"""
CLASSES = ['pass', 'missing_cap', 'crooked_cap', 'damaged_vial', 'no_vial']
DEFECTS = set(CLASSES[1:4])


def summarize(area, slots, expected_through=None, upstream=None):
    if area not in ('good', 'scrap'):
        raise ValueError('Unknown area')
    if len(slots) != 30 or {s['position'] for s in slots} != set(range(1, 31)):
        raise ValueError('Exactly 30 unique positions are required')
    if expected_through is not None and (type(expected_through) is not int or not 0 <= expected_through <= 30):
        raise ValueError('Expected placement count must be an integer from 0 to 30')
    upstream = upstream or {}
    present = [s['position'] for s in slots if s['presence'] is True]
    absent = [s['position'] for s in slots if s['presence'] is False]
    unknown = [s['position'] for s in slots if s['presence'] is None]
    last = max(present, default=0)
    boundary = max(last, expected_through or 0)
    holes = [p for p in absent if p <= boundary]
    pending = [p for p in absent if p > boundary]
    missing = True if holes else (None if any(p <= boundary for p in unknown) else False)
    # A last-slot drop is unobservable without a placement counter / order-complete signal.
    full = False if absent else (None if unknown else True)
    defect_slots = [s['position'] for s in slots if s['class'] in DEFECTS]
    uncertain = [s['position'] for s in slots if s['class'] is None]
    common = {
        'occupied': len(present), 'unknown_occupancy': len(unknown), 'capacity': 30,
        'last_observed_occupied_position': last, 'expected_through': expected_through,
        'fill_boundary': boundary,
        'boundary_source': 'placement_count_and_visible_occupancy' if expected_through is not None else 'visible_occupancy_lower_bound',
        'missing_positions': holes, 'not_yet_filled_positions': pending,
        'unknown_positions': uncertain, 'visible_defect_positions': defect_slots,
        'full': full, 'needs_review': any(s.get('review_status') != 'confirmed' for s in slots),
    }
    if area == 'good':
        bad = True if defect_slots else (None if uncertain else False)
        complete = False if full is False or bad is True else (True if full is True and bad is False else None)
        flags = {'complete_order': complete, 'missing_vial': missing, 'bad_misclassified_as_good': bad}
        common['in_progress'] = bool(pending) and missing is False and not defect_slots
    else:
        passing = [s['position'] for s in slots if s['class'] == 'pass']
        # upstream verdicts must refer to this frame and slot, not a tray-wide guess.
        confirmed_misroutes = [p for p in present if upstream.get(str(p)) == 'pass']
        unchecked = [p for p in present if upstream.get(str(p)) not in ('pass', 'reject')]
        routing = True if confirmed_misroutes else (None if unchecked or unknown else False)
        # Full is an operational exception even if routing is correct.
        no_problems = False if missing is True or full is True or routing is True else (
            True if missing is False and full is False and routing is False and not unknown else None)
        flags = {'no_problems': no_problems, 'missing_vial': missing,
                 'good_misclassified_as_bad': routing, 'full': full}
        common.update(visually_passing_scrap_positions=passing,
                      suspected_misroute_positions=[p for p in passing if upstream.get(str(p)) != 'reject'],
                      upstream_confirmed_misroute_positions=confirmed_misroutes,
                      upstream_unverified_positions=unchecked,
                      visible_pattern_ok=missing is False and not unknown)
    common['findings'] = flags
    common['positive_labels'] = [k for k, v in flags.items() if v is True]
    return common
