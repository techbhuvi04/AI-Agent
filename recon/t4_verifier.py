import pandas as pd

SUM_TOLERANCE = 0.50


def verify_claims(result_df, bank_df, claims):
    valid_claims = []
    rejected_claims = []

    bank_lookup = bank_df.set_index("utr")["credit"].to_dict()

    for claim in claims:
        utr = claim["credit_utr"]
        if utr not in bank_lookup:
            claim["rejection_reason"] = f"Unknown UTR: {utr}"
            rejected_claims.append(claim)
            continue

        target_credit = bank_lookup[utr]
        entry_ids = claim["proposed_entry_ids"]
        
        valid_indices = []
        invalid_indices = []
        for eid in entry_ids:
            if eid in result_df.index and pd.isna(result_df.at[eid, "assigned_utr"]):
                valid_indices.append(eid)
            else:
                invalid_indices.append(eid)
                
        if invalid_indices:
            claim["rejection_reason"] = f"Invalid or already assigned entry IDs: {invalid_indices}"
            rejected_claims.append(claim)
            continue
            
        if not valid_indices:
            claim["rejection_reason"] = "No valid entry IDs provided"
            rejected_claims.append(claim)
            continue

        subset_sum = result_df.loc[valid_indices, "net"].sum()
        
        if abs(subset_sum - target_credit) > SUM_TOLERANCE:
            claim["rejection_reason"] = f"Sum mismatch: target={target_credit:.2f}, subset_sum={subset_sum:.2f}"
            rejected_claims.append(claim)
            continue
            
        valid_claims.append(claim)

    return valid_claims, rejected_claims
