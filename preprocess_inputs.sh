#/bin/bash
#$ -S /bin/bash
set -x -e

#ROOT="/Users/sravikumar/Box Sync/PennPhD/Research/PICSL/exvivo_mtl_unet"
ROOT='/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet'


INPUTS="$ROOT/preproc_all_27"
DATADIR="$ROOT/data_csv"
CODEDIR="$ROOT/code"

SUBJ_TXT=$ROOT/subj_train_fold1.txt
IND_ALL="$(cat "$SUBJ_TXT")"

function main()
{

  mkdir -p $DATADIR
  process_inputs all
  #process_inputs train
  #process_inputs test
	
#  Preparation
#  PreparationTest

# PreparePulkitData
}

function process_inputs()
{

  mode=${1?}

  FILE=$ROOT/subj_${mode}.txt
  N=$(cat "$FILE" | wc -l | sed 's/^ *//g')

  for ((i=1;i<=$N;i++)); do

        LINE=$(cat "$FILE" | head -n $i | tail -n 1)
        id=$(echo $LINE | cut -d ' ' -f 1)

	echo $id

	WDIR=$ROOT/preproc_${mode}_${N}
	mkdir -p "$WDIR"
  	mkdir -p "$WDIR/mri" "$WDIR/dmap" "$WDIR/seg" "$WDIR/intermediate" "$WDIR/seg_nosrlm"

	#echo "Normalizing image for  $id"
	IMG=$INPUTS/${id}/${id}_clip_n4.nii.gz
	SEG=$INPUTS/${id}/${id}_multilabel_corrected_seg_whippo.nii.gz
	DMAP=$INPUTS/${id}/${id}_coords-IO.nii.gz
	SRLM_SEG=$INPUTS/${id}/${id}_axisalign_srlm_sr.nii.gz

 	IMG_TRIM=$WDIR/mri/${id}_trimmed_img.nii.gz
	SEG_TRIM=$WDIR/seg/${id}_trimmed_phg.nii.gz
	SEG_NOSRLM_TRIM=$WDIR/seg_nosrlm/${id}_trimmed_phg_nosrlm.nii.gz
	DMAP_TRIM=$WDIR/dmap/${id}_trimmed_dmap.nii.gz
	SEG_COMB=$WDIR/intermediate/${id}_multilabel_wsrlm.nii.gz

	#Combine srlm with multilabel seg
  	c3d "$SEG" -replace 4 5 1 7 -as MS "$SRLM_SEG" -add -replace 6 4 8 4 7 1 -o "$SEG_COMB"

      #Trim input image to only contain segmented region
	c3d "$SEG_COMB" -trim 0vox -o "$SEG_TRIM" -thresh -inf inf 1 0 -popas MASK "$IMG" \
	-push MASK -reslice-identity -as R "$IMG" -add -push R -times -trim 0vox \
        -shift -1 -o "$IMG_TRIM" \
	"$DMAP" -push R -add -push R -times -trim 0vox -shift -1 -o "$DMAP_TRIM"

	c3d "$SEG" -trim 0vox -o "$SEG_NOSRLM_TRIM" 

	#Downsample the trimmed input image since such a high res is not required. Patch will capture more info
  #	c3d $IMG_TRIM -resample 75% -o $INPUTS/${id}/${id}_downsample_img.nii.gz
  #	c3d $SEG_TRIM -resample 75% -o $INPUTS/${id}/${id}_downsample_phg.nii.gz
	# Post processing to visualize test results

  done
}


function Preparation()
{

  N=$(cat $SUBJ_TXT | wc -l)
  rm -rf $DATADIR/split.csv

  for ((i=1;i<=${N};i++)); do

    LINE=$(cat $SUBJ_TXT | head -n $i | tail -n 1)
    id=$(echo $LINE | cut -d ' ' -f 1)
    read dummmy type idint <<< $(cat $SUBJ_TXT | grep $id)

    IMG=$INPUTS/mri/${id}_trimmed_img.nii.gz
    SEG=$INPUTS/seg/${id}_trimmed_phg.nii.gz
    DMAP=$INPUTS/dmap/${id}_trimmed_dmap.nii.gz

    echo "$IMG,$SEG,$idint,"Control",$type,$DMAP" >> $DATADIR/split.csv

  done
}

# Jan 6 - got rid of test dataset. Can cross validate instead to make use of all data
function PreparationTest()
{

  N=$(cat $ROOT/subj_test.txt | wc -l)
  rm -rf $DATADIR/split_test.csv

  for ((i=1;i<=${N};i++)); do

    LINE=$(cat $ROOT/subj_test.txt | head -n $i | tail -n 1)
    id=$(echo $LINE | cut -d ' ' -f 1)
    read dummmy type idint <<< $(cat $ROOT/subj_test.txt | grep $id)

      IMG=$INPUTS/${id}/${id}_trimmed_img.nii.gz
      SEG=$INPUTS/${id}/${id}_trimmed_phg.nii.gz

      echo "$IMG,$SEG,$idint,"Control",$type" >> $DATADIR/split_test.csv

  done
}

function PreparePulkitData()
{

for filename in /home/sadhana-ravikumar/Documents/Sadhana/mtl94_forpulkit/*.nii.gz; do

  file=$(basename $filename)
  id=$(echo $file | cut -d_ -f1)
  echo "$filename,$filename,$id,"Control","test"" >> $DATADIR/test_pulkit.csv

 done

}
main
