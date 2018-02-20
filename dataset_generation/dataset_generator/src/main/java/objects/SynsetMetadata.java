package objects;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class SynsetMetadata implements Serializable {
	
	private static final long serialVersionUID = 1L;
	Integer synsetID; 
	List<Integer> domainIDs;
	List<Double> domainScores;
	List<Integer> domainSource;
	Integer mainImage; 
	List<Integer> images;
	Integer imageSource;
	Integer mainSense;
	List<Integer> Senses;
	List<Integer> definitions;
	List<Integer> defLangID;
	List<Integer> defSourceID;
	
	public SynsetMetadata() {
		super();
		domainIDs = new ArrayList<Integer>();
		domainScores = new ArrayList<Double>();
		domainSource = new ArrayList<Integer>();
		images  = new ArrayList<Integer>();
		Senses  = new ArrayList<Integer>();
		definitions  = new ArrayList<Integer>();
		defLangID  = new ArrayList<Integer>();
		defSourceID  = new ArrayList<Integer>();
	}
	
	public SynsetMetadata(Integer synsetID, List<Integer> domainIDs, List<Double> domainScores, List<Integer> domainSource,
			Integer mainImage, List<Integer> images, Integer imageSource, Integer mainSense, List<Integer> senses,
			List<Integer> definitions, List<Integer> defLangID, List<Integer> defSourceID) {
		super();
		this.synsetID = synsetID;
		this.domainIDs = domainIDs;
		this.domainScores = domainScores;
		this.domainSource = domainSource;
		this.mainImage = mainImage;
		this.images = images;
		this.imageSource = imageSource;
		this.mainSense = mainSense;
		Senses = senses;
		this.definitions = definitions;
		this.defLangID = defLangID;
		this.defSourceID = defSourceID;
	}
	public Integer getsynsetID() {
		return synsetID;
	}
	public void setSynsetID(Integer synsetID) {
		this.synsetID = synsetID;
	}
	public List<Integer> getDomainIDs() {
		return domainIDs;
	}
	public void setDomainIDs(List<Integer> domainIDs) {
		this.domainIDs = domainIDs;
	}
	public void addDomainID(Integer domainID){
		this.domainIDs.add(domainID);
	}
	public Boolean containsDomainID(Integer domainID){
		if (this.domainIDs.contains(domainID)){
			return true;
		}
		return false;
	}
	public List<Double> getDomainScores() {
		return domainScores;
	}
	public void setDomainScores(List<Double> domainScores) {
		this.domainScores = domainScores;
	}
	public void addDomainScore(Double domainSc){
		this.domainScores.add(domainSc);
	}
	public List<Integer> getDomainSource() {
		return domainSource;
	}
	public void setDomainSource(List<Integer> domainSource) {
		this.domainSource = domainSource;
	}
	public void addDomainSource(Integer domainSo){
		this.domainSource.add(domainSo);
	}
	public Integer getMainImage() {
		return mainImage;
	}
	public void setMainImage(Integer mainImage) {
		this.mainImage = mainImage;
	}
	public List<Integer> getImages() {
		return images;
	}
	public Boolean containsImage(Integer imageID){
		if (this.images.contains(imageID)){
			return true;
		}
		return false;
	}
	public void setImages(List<Integer> images) {
		this.images = images;
	}
	public void addImage(Integer imageID){
		this.images.add(imageID);
	}
	public Integer getImageSource() {
		return imageSource;
	}
	public void setImageSource(Integer imageSource) {
		this.imageSource = imageSource;
	}
	public Integer getMainSense() {
		return mainSense;
	}
	public void setMainSense(Integer mainSense) {
		this.mainSense = mainSense;
	}
	public List<Integer> getSenses() {
		return Senses;
	}
	public void setSenses(List<Integer> senses) {
		Senses = senses;
	}
	public void addSense(Integer senseID){
		this.Senses.add(senseID);
	}
	public List<Integer> getDefinitions() {
		return definitions;
	}
	public void setDefinitions(List<Integer> definitions) {
		this.definitions = definitions;
	}
	public Boolean containsDefinition(Integer defID){
		if (this.definitions.contains(defID)){
			return true;
		}
		return false;
	}
	public void addDefinitions(Integer defID){
		this.definitions.add(defID);
	}
	public List<Integer> getDefLangID() {
		return defLangID;
	}
	public void setDefLangID(List<Integer> defLangID) {
		this.defLangID = defLangID;
	}
	public void addDefLangID(Integer langID){
		this.defLangID.add(langID);
	}
	public List<Integer> getDefSourceID() {
		return defSourceID;
	}
	public void ListDefSourceID(List<Integer> defSourceID) {
		this.defSourceID = defSourceID;
	}
	public void addDefSourceID(Integer defSource){
		this.defSourceID.add(defSource);
	}
	@Override
	public String toString() {
		return "SynListMetadata [synsetID=" + synsetID + ", domainIDs=" + domainIDs + ", domainScores=" + domainScores
				+ ", domainSource=" + domainSource + ", mainImage=" + mainImage + ", images=" + images
				+ ", imageSource=" + imageSource + ", mainSense=" + mainSense + ", Senses=" + Senses + ", definitions="
				+ definitions + ", defLangID=" + defLangID + ", defSourceID=" + defSourceID + "]";
	}
}
