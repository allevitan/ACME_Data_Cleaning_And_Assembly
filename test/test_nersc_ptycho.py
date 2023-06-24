from acme_data_cleaning import nersc


if __name__ == "__main__":
    cxiname = "NS_230216033_ccdframes_0_0.cxi"

    nersc.ptycho(
        cxiname=cxiname,
    )
