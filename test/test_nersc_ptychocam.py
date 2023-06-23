from acme_data_cleaning import nersc


if __name__ == "__main__":
    cxiname = "NS_230216033_ccdframes_0_0.cxi"
    n_iter = 2000
    period_illu_refine = 1
    period_bg_refine = 1
    use_illu_mask = True

    nersc.ptychocam(
        cxiname=cxiname,
        n_iter=n_iter,
        period_illu_refine=period_illu_refine,
        period_bg_refine=period_bg_refine,
        use_illu_mask=use_illu_mask
    )
