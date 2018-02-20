package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="wordPOS")
public class WordPOS {
	private Long wordPOS_id;
	private String wordPOS_value;
	
	
	public WordPOS() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public WordPOS(Long wordPOS_id, String wordPOS_value) {
		super();
		this.wordPOS_id = wordPOS_id;
		this.wordPOS_value = wordPOS_value;
	}
	
	@Id
	@Column(name = "wordPOS_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getWordPOS_id() {
		return wordPOS_id;
	}

	public void setWordPOS_id(Long wordPOS_id) {
		this.wordPOS_id = wordPOS_id;
	}

	public String getWordPOS_value() {
		return wordPOS_value;
	}

	public void setWordPOS_value(String wordPOS_value) {
		this.wordPOS_value = wordPOS_value;
	}

	@Override
	public String toString() {
		return "WordPOS [wordPOS_id=" + wordPOS_id + ", wordPOS_value=" + wordPOS_value + "]";
	}
	
	
	
}
