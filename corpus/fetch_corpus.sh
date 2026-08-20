#!/usr/bin/env bash
# corpus/fetch_corpus.sh — recupere les PDF sources du corpus agronomique.
#
# GENERE depuis corpus/sources.yaml par l'outillage de collecte (non redistribue, meme
# regime que corpus/txt/fetched/) — ne pas editer a la main : toute correction faite ici
# est perdue a la prochaine generation, elle doit se faire dans sources.yaml.
#
# Pourquoi ce script existe : les documents institutionnels du Sahel (RECA Niger,
# INRAN, ICRISAT-fr) ne declarent AUCUNE licence. Nous ne redistribuons donc pas
# leur texte. Ce script les recupere chez l'editeur, comme download_model.sh
# recupere les poids chez HuggingFace : la regle « 100 % offline » porte sur
# l'INFERENCE, pas sur le setup — sinon download_model.sh la violerait aussi.
#
# Proprietes : idempotent · aucun credential · taille attendue verifiee.
#
# Usage :
#   bash corpus/fetch_corpus.sh              # tout
#   bash corpus/fetch_corpus.sh fetched      # seulement les sources non redistribuables
#   bash corpus/fetch_corpus.sh committed    # seulement le socle CC BY
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pdf"
mkdir -p "$DEST"

FILTRE="${1:-tout}"
OK=0
IGNORES=0
ECHECS=0

# Telecharge $1 vers $2 en verifiant que la taille finale vaut $3 octets.
# Un fichier tronque est pire qu'un fichier absent : il produit une extraction
# partielle qu'on prendrait pour un document complet.
recupere() {
  local url="$1" nom="$2" attendu="$3" regime="$4"
  local cible="$DEST/$nom"

  if [ "$FILTRE" != "tout" ] && [ "$FILTRE" != "$regime" ]; then
    return 0
  fi

  if [ -f "$cible" ]; then
    local actuel
    actuel=$(wc -c < "$cible" | tr -d ' ')
    if [ "$actuel" = "$attendu" ]; then
      echo "  = deja present et complet : $nom"
      IGNORES=$((IGNORES + 1))
      return 0
    fi
    echo "  ! taille inattendue ($actuel != $attendu), nouvelle tentative : $nom"
  fi

  echo "  > $nom ($attendu octets)"
  if ! curl -fsSL --retry 5 --retry-delay 5 -C - -o "$cible" "$url"; then
    echo "  X ECHEC de telechargement : $url" >&2
    ECHECS=$((ECHECS + 1))
    return 0
  fi

  local final
  final=$(wc -c < "$cible" | tr -d ' ')
  if [ "$final" != "$attendu" ]; then
    echo "  X TAILLE INCORRECTE : $nom vaut $final, attendu $attendu" >&2
    ECHECS=$((ECHECS + 1))
    return 0
  fi
  OK=$((OK + 1))
}

echo "== corpus : recuperation des PDF (filtre = $FILTRE) =="
recupere "https://reca-niger.org/IMG/pdf/recueil_de_fiches_techniques_validees_vf.pdf" "recueil_de_fiches_techniques_validees_vf.pdf" 4595072 fetched
recupere "https://reca-niger.org/IMG/pdf/fiche_technique_niebe_dan_hadjia.pdf" "fiche_technique_niebe_dan_hadjia.pdf" 232729 fetched
recupere "https://reca-niger.org/IMG/pdf/fiche_technique_pour_le_repiquage_du_mil.pdf" "fiche_technique_pour_le_repiquage_du_mil.pdf" 246154 fetched
recupere "https://reca-niger.org/IMG/pdf/note_conseil_foreur_tiges_de_mil_coniesta_ignefusalis_2024.pdf" "note_conseil_foreur_tiges_de_mil_coniesta_ignefusalis_2024.pdf" 404977 fetched
recupere "https://reca-niger.org/IMG/pdf/juin_-_juillet_selection_thematique_duddal-presse_maladie_des_cultures_vivrieres.pdf" "juin_-_juillet_selection_thematique_duddal-presse_maladie_des_cultures_vivrieres.pdf" 1295258 fetched
recupere "http://oar.icrisat.org/id/document/48470" "MONTAGE DU PROJET PASAM-TAI WEB.pdf" 4732727 fetched
recupere "http://oar.icrisat.org/id/document/57526" "Agronomy_15_6_1-16_2025.pdf" 1935708 committed
recupere "http://oar.icrisat.org/id/document/57444" "PLoS One_20_5_1-26_2025.pdf" 3725569 committed
recupere "http://oar.icrisat.org/id/document/57495" "1f010dfe-0ae0-4a04-8fee-6b21066a7ae5.pdf" 2980774 committed
recupere "http://oar.icrisat.org/id/document/57215" "Journal of Food Quality_2025_1-21_2025.pdf" 676917 committed
recupere "http://oar.icrisat.org/id/document/56273" "International Journal of Digital Earth_17_1_1-18_2024.pdf" 2494018 committed
recupere "http://oar.icrisat.org/id/document/57507" "Frontiers in Climate_6_01-12_2025.pdf" 1829744 committed
recupere "http://oar.icrisat.org/id/document/56942" "PLOS ONE_19_11_1-35_2024.pdf" 4915704 committed
recupere "http://oar.icrisat.org/id/document/56309" "Physiologia Plantarum_176_3_1-20_2024.pdf" 19383753 fetched
recupere "http://oar.icrisat.org/id/document/57100" "Journal of Applied Entomology_1-21_2025.pdf" 944507 committed
recupere "http://oar.icrisat.org/id/document/57372" "International Journal of Tropical Insect Science_45_593-600_2025.pdf" 992670 fetched
recupere "http://oar.icrisat.org/id/document/57370" "Soil and Tillage Research_246_1-9_2025.pdf" 2345917 committed
recupere "http://oar.icrisat.org/id/document/56131" "Frontiers in Climate_06_01-15_2024.pdf" 6274611 committed
recupere "http://oar.icrisat.org/id/document/57448" "Agronomy for Sustainable Development_44_1-16_2024.pdf" 2437921 committed
recupere "http://oar.icrisat.org/id/document/56229" "elife_12_1-21_2024.pdf" 5402965 committed
recupere "https://reca-niger.org/IMG/pdf/catalogue_des_especes_et_varietes_niger.pdf" "catalogue_des_especes_et_varietes_niger.pdf" 6680246 fetched
recupere "https://reca-niger.org/IMG/pdf/les_varietes_de_mil_2020.pdf" "les_varietes_de_mil_2020.pdf" 546189 fetched
recupere "https://reca-niger.org/IMG/pdf/varietes_de_semences_certifiees_disponibles_2025_reca.pdf" "varietes_de_semences_certifiees_disponibles_2025_reca.pdf" 518753 fetched
recupere "https://reca-niger.org/IMG/pdf/manuel_niebe_a5_dec_2022.pdf" "manuel_niebe_a5_dec_2022.pdf" 5581144 fetched
recupere "https://reca-niger.org/IMG/pdf/Spodoptera_mais_fiche_technique_2017.pdf" "Spodoptera_mais_fiche_technique_2017.pdf" 953759 fetched
recupere "https://reca-niger.org/IMG/pdf/fiche_descriptive_frugiperda.pdf" "fiche_descriptive_frugiperda.pdf" 806447 fetched
recupere "https://reca-niger.org/IMG/pdf/guide_identification_mem_lamine.pdf" "guide_identification_mem_lamine.pdf" 548506 fetched
recupere "https://reca-niger.org/IMG/pdf/foreurs_de_tiges_inran.pdf" "foreurs_de_tiges_inran.pdf" 369269 fetched
recupere "https://reca-niger.org/IMG/pdf/sorgho_ssd-35.pdf" "sorgho_ssd-35.pdf" 282509 fetched
recupere "https://reca-niger.org/IMG/pdf/Presentation_fiche_conseil2013.pdf" "Presentation_fiche_conseil2013.pdf" 591375 fetched
recupere "https://reca-niger.org/IMG/pdf/Fiche_conseil_Lambda_cyhalothrine_Version_22septembre2013.pdf" "Fiche_conseil_Lambda_cyhalothrine_Version_22septembre2013.pdf" 546951 fetched
recupere "https://reca-niger.org/IMG/pdf/conseil_en_fertilisation_note_2_reca.pdf" "conseil_en_fertilisation_note_2_reca.pdf" 476801 fetched
recupere "https://reca-niger.org/IMG/pdf/FT_Bandes_niebe_mil_INRAN_2015.pdf" "FT_Bandes_niebe_mil_INRAN_2015.pdf" 544103 fetched
recupere "https://reca-niger.org/IMG/pdf/gdt_fiches_techniques_2022_sacs_pics.pdf" "gdt_fiches_techniques_2022_sacs_pics.pdf" 546596 fetched
recupere "https://reca-niger.org/IMG/pdf/FT_Compostage_aerien_INRAN.pdf" "FT_Compostage_aerien_INRAN.pdf" 640377 fetched
recupere "https://reca-niger.org/IMG/pdf/vca4d.niger_.arachide.28112023_approved_v1.pdf" "vca4d.niger_.arachide.28112023_approved_v1.pdf" 5484324 fetched
recupere "https://reca-niger.org/IMG/pdf/RECA_appui-conseil_Note7_Unions_producteurs_niebe_Burkina_2012.pdf" "RECA_appui-conseil_Note7_Unions_producteurs_niebe_Burkina_2012.pdf" 464028 fetched
recupere "https://reca-niger.org/IMG/pdf/2014-09_fermeture_et_liberation_des_champs.pdf" "2014-09_fermeture_et_liberation_des_champs.pdf" 217162 fetched
recupere "https://reca-niger.org/IMG/pdf/note_effets_pratiques_recommandees_cgef_rendements_cultures__zinder_vf.pdf" "note_effets_pratiques_recommandees_cgef_rendements_cultures__zinder_vf.pdf" 475880 fetched

echo "== bilan : $OK telecharge(s), $IGNORES deja present(s), $ECHECS echec(s) =="
if [ "$ECHECS" -gt 0 ]; then
  echo "Certaines sources n'ont pas pu etre recuperees. Le corpus `committed`" >&2
  echo "suffit a faire fonctionner le RAG : voir REPORT.md." >&2
  exit 1
fi
